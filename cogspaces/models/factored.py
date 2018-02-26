from math import ceil, floor

import tempfile
import warnings

import torch
from cogspaces.optim.lbfgs import LBFGSScipy
from cogspaces.data import ImgContrastDataset, RepeatedDataLoader
from sklearn.base import BaseEstimator
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear, Dropout, init
from torch.nn.functional import mse_loss, nll_loss
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np


class LinearAutoEncoder(nn.Module):
    def __init__(self, in_features, out_features,
                 l1_penalty=0., l2_penalty=0.):
        super().__init__()
        self.linear = Linear(in_features, out_features, bias=False)
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, input):
        return self.linear(input)

    def reconstruct(self, projection):
        weight = self.linear.weight
        # gram = torch.matmul(weight.transpose(0, 1), weight)
        rec = torch.matmul(projection, weight)
        # rec, LU = torch.gesv(rec, gram)
        return rec

    def penalty(self):
        penalty = self.l2_penalty * .5 * torch.sum(self.linear.weight ** 2)
        penalty += self.l1_penalty * torch.sum(torch.abs(self.linear.weight))
        return penalty


class MultiClassifierHead(nn.Module):
    def __init__(self, in_features,
                 target_sizes, l1_penalty=0., l2_penalty=0.):
        super().__init__()
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
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

    def penalty(self):
        penalty = 0
        for study, classifier in self.classifiers.items():
            penalty += self.l2_penalty * .5 * torch.sum(classifier.weight ** 2)
            penalty += self.l1_penalty * torch.sum(
                torch.abs(classifier.weight))
        return penalty


class MultiClassifierModule(nn.Module):
    def __init__(self, in_features, embedding_size,
                 target_sizes, dropout=0., input_dropout=0.,
                 l1_penalty=0., l2_penalty=0.):
        super().__init__()
        if embedding_size == 'auto':
            embedding_size = sum(target_sizes.values())
        self.embedder = LinearAutoEncoder(in_features,
                                          embedding_size,
                                          l1_penalty=l1_penalty,
                                          l2_penalty=l2_penalty)
        self.dropout = Dropout(dropout)
        self.input_dropout = Dropout(input_dropout)
        self.classifier_head = MultiClassifierHead(embedding_size,
                                                   target_sizes,
                                                   l1_penalty=l1_penalty,
                                                   l2_penalty=l2_penalty)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedder.reset_parameters()
        self.classifier_head.reset_parameters()

    def forward(self, input):
        embeddings = {}
        for study, sub_input in input.items():
            embeddings[study] = self.dropout(
                self.embedder(self.input_dropout(sub_input)))
        return self.classifier_head(embeddings)

    def reconstruct(self, input):
        recs = {}
        for study, sub_input in input.items():
            recs[study] = self.embedder.reconstruct(self.embedder(sub_input))
        return recs

    def penalty(self):
        penalty = self.classifier_head.penalty() + self.embedder.penalty()
        return penalty


class FactoredClassifier(BaseEstimator):
    def __init__(self, embedding_size=50,
                 batch_size=128, optimizer='adam',
                 dropout=0.5,
                 input_dropout=0.25,
                 l2_penalty=0.,
                 l1_penalty=0.,
                 autoencoder_loss=0.,
                 max_iter=10000, verbose=0,
                 device=-1):
        self.embedding_size = embedding_size
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = device
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.autoencoder_loss = autoencoder_loss

    def fit(self, X, y, study_weights=None,
            callback=None):
        cuda, device = self._check_cuda()

        data = next(iter(X.values()))
        in_features = data.shape[1]

        data_loaders = {}
        target_sizes = {}

        n_samples = sum(len(this_X) for this_X in X.values())

        if study_weights is None:
            study_weights = {study: 1. for study in X}

        for study in X:
            target_sizes[study] = int(y[study].max()) + 1
            if self.optimizer == 'adam':
                data_loaders[study] = RepeatedDataLoader(
                    ImgContrastDataset(X[study], y[study]),
                    batch_size=self.batch_size, pin_memory=cuda)
            elif self.optimizer == 'lbfgs':
                if self.dropout > 0.:
                    raise ValueError('Dropout should not be used'
                                     'with LBFGS solver.')
                data_loaders[study] = DataLoader(
                    ImgContrastDataset(X[study], y[study]),
                    batch_size=len(X[study]), pin_memory=cuda)
            else:
                raise ValueError

        self.module_ = MultiClassifierModule(
            in_features=in_features,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            embedding_size=self.embedding_size,
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            target_sizes=target_sizes)
        report_every = ceil(self.max_iter / self.verbose)

        if self.optimizer == 'lbfgs':
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
                        study_loss += nll_loss(preds, contrasts,
                                               size_average=False)
                        if self.autoencoder_loss > 0:
                            recs = self.module_.reconstruct({study: data})[
                                study]
                            study_loss += mse_loss(data, recs,
                                                   size_average=False) \
                                * self.autoencoder_loss
                        n_samples += len(preds)
                    loss += study_loss * study_weights[study] / n_samples
                loss += self.module_.penalty()
                return loss

            self.optimizer_ = LBFGSScipy(self.module_.parameters(),
                                         callback=callback,
                                         max_iter=self.max_iter,
                                         tolerance_grad=0,
                                         tolerance_change=0,
                                         report_every=report_every)
            self.optimizer_.step(closure)
            self.n_iter_ = self.optimizer_.n_iter_
        elif self.optimizer == 'adam':
            data_loaders = {study: iter(loader) for study, loader in
                            data_loaders.items()}

            self.optimizer_ = Adam(self.module_.parameters())

            total_seen_samples = 0
            mean_loss = 0
            seen_samples = 0
            old_epoch = -1
            self.n_iter_ = 0
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
                    loss = nll_loss(preds, contrasts)
                    if self.autoencoder_loss > 0:
                        recs = self.module_.reconstruct({study: data})[study]
                        loss += mse_loss(data, recs) * self.autoencoder_loss

                    loss *= study_weights[study]
                    loss += self.module_.penalty()
                    loss.backward()
                    self.optimizer_.step()

                    mean_loss += loss.data[0]
                    seen_samples += len(data)
                    total_seen_samples += len(data)
                    self.n_iter_ = total_seen_samples / n_samples
                    epoch = floor(self.n_iter_)
                    if report_every is not None and epoch > old_epoch \
                            and epoch % report_every == 0:
                        mean_loss = mean_loss / seen_samples
                        print('Epoch %.2f, train loss: % .4f' %
                              (self.n_iter_, mean_loss))
                        mean_loss = 0
                        seen_samples = 0

                        if callback is not None:
                            callback(self.n_iter_)
                    old_epoch = epoch
        else:
            raise NotImplementedError('Optimizer not supported.')

    def _check_cuda(self):
        if self.device > -1 and not torch.cuda.is_available():
            warnings.warn('Cuda is not available on this system: computation'
                          'will be made on CPU.')
            device = -1
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
        return {study: np.argmax(pred, axis=1) for study, pred in
                preds.items()}

    @property
    def coef_(self):
        coefs = {}
        for study, classifier in \
                self.module_.classifier_head.classifiers.items():
            coef = classifier.weight.data
            coef = torch.matmul(coef, self.module_.embedder.linear.weight.data)
            coef = coef.transpose(0, 1)
            coefs[study] = coef.cpu().numpy()
        return coefs

    @property
    def intercept_(self):
        return {study: classifier[study].intercept.data.cpu().numpy()
                for study, classifier in
                self.module_.classifier_head.classifiers.items()}

    @property
    def coef_cat_(self):
        return np.concatenate(list(self.coef_.values()), axis=1)

    @property
    def intercept_cat_(self):
        return np.concatenate(self.intercept_)

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
