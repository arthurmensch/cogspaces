import tempfile
import warnings

import torch
from cogspaces.optim.lbfgs import LBFGSScipy
from cogspaces.data import ImgContrastDataset, RepeatedDataLoader
from sklearn.base import BaseEstimator
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear, Dropout, NLLLoss, init
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np


class MultiClassifierHead(nn.Module):
    def __init__(self, in_features,
                 target_sizes):
        super().__init__()
        self.classifiers = {}
        for study, size in target_sizes.items():
            self.classifiers[study] = Linear(in_features, size)
            self.add_module('classifier_%s' % study, self.classifiers[study])
        self.softmax = nn.LogSoftmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        for classifier in self.classifiers.values():
            init.xavier_uniform(classifier.weight)
            classifier.bias.data.fill_(0.)

    def forward(self, input):
        preds = {}
        for study, sub_input in input.items():
            pred = self.softmax(self.classifiers[study](sub_input))
            preds[study] = pred
        return preds


class MultiClassifierModule(nn.Module):
    def __init__(self, in_features, embedding_size,
                 target_sizes, dropout=0., input_dropout=0.):
        super().__init__()
        self.embedder = Linear(in_features, embedding_size, bias=False)
        self.dropout = Dropout(dropout)
        self.input_dropout = Dropout(input_dropout)
        self.classifier_head = MultiClassifierHead(embedding_size,
                                                   target_sizes)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform(self.embedder.weight)
        self.classifier_head.reset_parameters()

    def forward(self, input):
        embeddings = {}
        for study, sub_input in input.items():
            embeddings[study] = self.dropout(
                self.embedder(self.input_dropout(sub_input)))
        return self.classifier_head(embeddings)


class FactoredClassifier(BaseEstimator):
    def __init__(self, embedding_size=50,
                 batch_size=128, optimizer='adam',
                 dropout=0.5,
                 input_dropout=0.25,
                 l2_penalty=0.,
                 max_iter=10000, report_every=None, device=-1):
        self.embedding_size = embedding_size
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.report_every = report_every
        self.device = device
        self.l2_penalty = l2_penalty

    def fit(self, X, y, X_val=None, y_val=None):
        cuda, device = self._check_cuda()

        data = next(iter(X.values()))
        in_features = data.shape[1]

        data_loaders = {}
        target_sizes = {}

        for study in X:
            target_sizes[study] = int(y[study].max()) + 1
            if self.optimizer == 'adam':
                data_loaders[study] = RepeatedDataLoader(
                    ImgContrastDataset(X[study], y[study]),
                    batch_size=self.batch_size, pin_memory=cuda)
            elif self.optimizer == 'lbfgs':
                if self.dropout > .5:
                    raise ValueError('Dropout should not be used'
                                     'with LBFGS solver.')
                data_loaders[study] = DataLoader(
                    ImgContrastDataset(X[study], y[study]),
                    batch_size=len(X[study]), pin_memory=cuda)
            else:
                raise ValueError

        if X_val is not None and y_val is not None:
            eval_data_loaders = {}
            for study in X_val:
                eval_data_loaders[study] = DataLoader(
                    ImgContrastDataset(X_val[study], y_val[study]),
                    shuffle=False, batch_size=128,
                    pin_memory=cuda)
        else:
            eval_data_loaders = None

        loss_function = NLLLoss()
        self.module_ = MultiClassifierModule(in_features=in_features,
                                             dropout=self.dropout,
                                             input_dropout=self.input_dropout,
                                             embedding_size=self.embedding_size,
                                             target_sizes=target_sizes)

        if self.optimizer == 'lbfgs':
            def callback():
                if eval_data_loaders is not None:
                    accuracies = self._score(eval_data_loaders)
                    print('Test accuracies:')
                    print(accuracies)
                    W = self.classification_matrix()
                    if cuda:
                        W = W.cpu()
                    rank = np.linalg.matrix_rank(W.numpy())
                    print('Rank :', rank)

            def closure():
                self.module_.train()
                loss = 0
                for study, loader in data_loaders.items():
                    study_loss = 0
                    n_samples = 0
                    for data, contrasts in loader:
                        contrasts = contrasts.squeeze()
                        if cuda:
                            data = data.cuda(device=device)
                            contrasts = contrasts.cuda(device=device)
                        data = Variable(data)
                        contrasts = Variable(contrasts)
                        preds = self.module_({study: data})[study]
                        study_loss += loss_function(preds,
                                                    contrasts) * len(preds)
                        n_samples += len(preds)
                    loss += study_loss / n_samples
                    loss += .5 * self.l2_penalty * torch.sum(
                        self.module_.classifier_head.
                        classifiers[study].weight ** 2)
                loss += .5 * self.l2_penalty * torch.sum(
                    self.module_.embedder.weight ** 2)
                return loss

            self.optimizer_ = LBFGSScipy(self.module_.parameters(),
                                         callback=callback,
                                         max_iter=self.max_iter,
                                         tolerance_grad=0,
                                         tolerance_change=0,
                                         report_every=self.report_every)
            self.optimizer_.step(closure)
        else:
            data_loaders = {study: iter(loader) for study,
                                                    loader in
                            data_loaders.items()}

            self.optimizer_ = Adam(self.module_.parameters(),
                                   weight_decay=self.l2_penalty)
            self.n_iter_ = 0
            mean_loss = 0
            n_samples = 0
            while self.n_iter_ < self.max_iter:
                for study, loader in data_loaders.items():
                    self.module_.train()
                    self.optimizer_.zero_grad()
                    data, contrasts = next(loader)
                    contrasts = contrasts.squeeze()
                    if cuda:
                        data = data.cuda(device=device)
                        contrasts = contrasts.cuda(device=device)
                    data = Variable(data)
                    contrasts = Variable(contrasts)
                    preds = self.module_({study: data})[study]
                    loss = loss_function(preds, contrasts)
                    loss.backward()
                    self.optimizer_.step()

                    mean_loss += loss.data[0]
                    n_samples += len(data)
                    self.n_iter_ += 1
                    if self.report_every is not None and \
                        self.n_iter_ % self.report_every == 0:
                        mean_loss = mean_loss / n_samples
                        print('Iteration %i,'
                              ' train loss: % .4f' % (self.n_iter_, mean_loss))
                        mean_loss = 0
                        n_samples = 0

                        if eval_data_loaders is not None:
                            accuracies = self._score(eval_data_loaders)
                            print('Test accuracies:')
                            print(accuracies)
                            W = self.classification_matrix()
                            if cuda:
                                W = W.cpu()
                            rank = np.linalg.matrix_rank(W.numpy())
                            print('Rank :', rank)

    def classification_matrix(self):
        classification_weights = []
        for study, classifier in self.module_.\
                classifier_head.classifiers.items():
            classification_weights.append(
                self.module_.classifier_head.classifiers[study].weight.data)
        classification_weights = torch.cat(classification_weights)
        return torch.matmul(self.module_.embedder.weight.data.transpose(0, 1),
                            classification_weights)

    def _check_cuda(self):
        if self.device > -1 and not torch.cuda.is_available():
            warnings.warn('Cuda is not available on this system: computation'
                          'will be made on CPU.')
            device = 0
            cuda = False
        else:
            device = self.device
            cuda = device > -1
        return cuda, device

    def _predict_proba(self, data_loaders):
        cuda, device = self._check_cuda()

        preds = {}
        for study, loader in data_loaders.items():
            pred = []
            for (data, _) in loader:
                if cuda:
                    data = data.cuda(device=device)
                data = Variable(data, volatile=True)
                self.module_.eval()
                pred.append(self.module_({study: data})[study])
            preds[study] = pred
        return preds

    def _score(self, data_loaders):
        preds = self._predict_proba(data_loaders)
        accuracies = {}
        for study in preds:
            accuracy = 0
            n_samples = 0
            for (data, truth), pred in zip(data_loaders[study], preds[study]):
                truth = truth.squeeze()
                _, pred = torch.max(pred, dim=1)
                truth = Variable(truth, volatile=True)
                accuracy += torch.sum((pred == truth).long())
                n_samples += len(truth)
            accuracies[study] = accuracy.data[0] / n_samples
        return accuracies

    def predict_proba(self, X):
        cuda, device = self._check_cuda()

        data_loaders = {}
        for study, this_X in X.items():
            data_loaders[study] = DataLoader(ImgContrastDataset(this_X),
                                             batch_size=len(this_X),
                                             shuffle=False,
                                             pin_memory=cuda)
        preds = self._predict_proba(data_loaders)
        preds = {study: torch.cat(pred) for study, pred in preds.items()}
        if cuda:
            preds = {study: pred.cpu() for study, pred in preds.items()}
        preds = {study: pred.data.numpy() for study, pred in preds.items()}
        return preds

    def predict(self, X):
        preds = self.predict_proba(X)
        return {study: np.argmax(pred, axis=1) for study, pred in preds.items()}

    def score(self, X, y):
        preds = self.predict(X)
        accuracies = {}
        for study, this_y in y.items():
            accuracies[study] = np.mean(preds[study] == this_y)
        return accuracies

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['module_', 'optimizer_']:
            if key in state:
                val = state.pop(key)
                with tempfile.SpooledTemporaryFile() as f:
                    torch.save(val, f)
                    f.seek(0)
                    state[key] = f.read()

        return state

    def __setstate__(self, state):
        disable_cuda = False
        for key in ['module_', 'optimizer_']:
            if key not in state:
                continue
            dump = state.pop(key)
            with tempfile.SpooledTemporaryFile() as f:
                f.write(dump)
                f.seek(0)
                if state['device'] > - 1 and not torch.cuda.is_available():
                    val = torch.load(
                        f, map_location=lambda storage, loc: storage)
                    disable_cuda = True
                else:
                    val = torch.load(f)
            state[key] = val
        if disable_cuda:
            warnings.warn(
                "Model configured to use CUDA but no CUDA devices "
                "available. Loading on CPU instead.")
            state['device'] = -1

        self.__dict__.update(state)
