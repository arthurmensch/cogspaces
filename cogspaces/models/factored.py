import tempfile
import warnings
from math import ceil, floor, sqrt

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear
from torch.nn.functional import nll_loss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cogspaces.data import NiftiTargetDataset, RepeatedDataLoader
from cogspaces.optim.lbfgs import LBFGSScipy


class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

    def backward(self, grad):
        return - grad

    def reset_parameters(self):
        pass


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class MultiTaskModule(nn.Module):
    def __init__(self, in_features,
                 shared_embedding_size,
                 private_embedding_size,
                 target_sizes,
                 input_dropout=0.,
                 dropout=0.,
                 activation='linear',
                 shared_embedding='hard+adversarial',
                 skip_connection=False):
        super().__init__()

        self.in_features = in_features
        self.shared_embedding_size = shared_embedding_size
        self.private_embedding_size = private_embedding_size

        if activation == 'linear':
            self.activation = Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)
        self.input_dropout = nn.Dropout(p=input_dropout)

        self.skip_connection = skip_connection
        self.shared_embedding = shared_embedding

        if private_embedding_size > 0:
            self.private_embedders = {}

        if shared_embedding_size > 0:

            def get_shared_embedder():
                return nn.Sequential(
                    Linear(in_features, shared_embedding_size, bias=False),
                    self.activation,
                    self.dropout)

            if 'hard' in shared_embedding:
                shared_embedder = get_shared_embedder()
                self.add_module('shared_embedder', shared_embedder)
                self.shared_embedders = {study: shared_embedder
                                         for study in target_sizes}
            else:
                self.shared_embedders = {study: get_shared_embedder()
                                         for study in target_sizes}
                for study in target_sizes:
                    self.add_module('shared_embedder_%s' % study,
                                    self.shared_embedders[study])

        self.classifiers = {}

        for study, size in target_sizes.items():
            if private_embedding_size > 0:
                self.private_embedders[study] = nn.Sequential(
                    Linear(in_features, private_embedding_size, bias=False),
                    self.activation,
                    self.dropout)
                self.add_module('private_embedder_%s' % study,
                                self.private_embedders[study])

            self.classifiers[study] = nn.Sequential(
                Linear(shared_embedding_size + private_embedding_size
                       + in_features * skip_connection, size),
                nn.LogSoftmax(dim=1))
            self.add_module('classifier_%s' % study, self.classifiers[study])

        if 'adversarial' in shared_embedding:
            self.study_classifier = nn.Sequential(
                Linear(shared_embedding_size, len(target_sizes), bias=True),
                nn.ReLU(),
                Linear(len(target_sizes), len(target_sizes), bias=True),
                nn.LogSoftmax(dim=1))
            self.gradient_reversal = GradientReversal()
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.ndimension() == 2:
                nn.init.xavier_uniform(param)
            elif param.ndimension() == 1:
                param.data.fill_(0.)

    def forward(self, input):
        preds = {}
        for study, sub_input in input.items():
            embedding = []

            if self.skip_connection:
                embedding.append(sub_input)

            sub_input = self.input_dropout(sub_input)

            if self.shared_embedding_size > 0:
                shared_embedding = self.shared_embedders[study](sub_input)
                if 'adversarial' in self.shared_embedding:
                    # adversarial
                    study_pred = self.study_classifier(
                        self.gradient_reversal(shared_embedding))
                else:
                    study_pred = Variable(
                        sub_input.data.new(sub_input.shape[0], 1),
                        requires_grad=False)
                embedding.append(shared_embedding)
            else:
                # Return dummy variable
                study_pred = Variable(
                    sub_input.data.new(sub_input.shape[0], 1),
                    requires_grad=False)

            if self.private_embedding_size > 0:
                private_embedding = self.private_embedders[study](sub_input)
                embedding.append(private_embedding)

            if self.private_embedding_size > 0 and \
                    self.shared_embedding_size > 0:
                corr = torch.bmm(private_embedding[:, :, None],
                                 shared_embedding[:, None, :])
                corr /= sqrt(self.private_embedding_size
                             * self.shared_embedding_size)
                penalty = torch.sum(corr ** 2)
            else:
                penalty = Variable(torch.FloatTensor([0.]))

            embedding = torch.cat(embedding, dim=1)
            pred = self.classifiers[study](embedding)

            preds[study] = study_pred, pred, penalty
        return preds

    def coefs(self):
        coefs = {}
        for study in self.classifiers:
            coef = self.classifiers[study][0].weight.data
            if self.skip_connection:
                skip_coef = coef[:, :self.in_features]
                if self.in_features < coef.shape[1]:
                    coef = coef[:, self.in_features:]
            else:
                skip_coef = 0
            if self.shared_embedding_size > 0:
                shared_coef = coef[:, :self.shared_embedding_size]
                shared_embed = self.shared_embedders[study][0].weight.data
                shared_coef = torch.matmul(shared_coef, shared_embed)
            else:
                shared_coef = 0
            if self.private_embedding_size > 0:
                private_coef = coef[:, -self.private_embedding_size:]
                private_embed = self.private_embedders[study][0].weight.data
                private_coef = torch.matmul(private_coef, private_embed)
            else:
                private_coef = 0

            coef = skip_coef + shared_coef + private_coef
            coefs[study] = coef.transpose(0, 1)
        return coefs

    def intercepts(self):
        return {study: classifier[0].bias.data for study, classifier
                in self.classifiers.items()}


class MultiTaskLoss(nn.Module):
    def __init__(self, study_weights=None, adversarial=True):
        super().__init__()
        self.study_weights = study_weights
        self.adversarial = adversarial

    def forward(self, inputs, targets):
        loss = 0
        for study in inputs:
            study_pred, pred, penalty = inputs[study]
            study_target, target = targets[study][:, 0], targets[study][:, 1]
            if self.adversarial:
                study_loss = nll_loss(study_pred, study_target,
                                      size_average=False) * 10
            else:
                study_loss = Variable(torch.Tensor([0.]))
            pred_loss = nll_loss(pred, target, size_average=False)
            
            loss += (penalty + study_loss + pred_loss) * self.study_weights[study]
        return loss


class FactoredClassifier(BaseEstimator):
    def __init__(self, skip_connection=False, shared_embedding_size=30,
                 private_embedding_size=5,
                 activation='linear',
                 shared_embedding='hard+adversarial',
                 batch_size=128, optimizer='sgd',
                 lr=0.001,
                 dropout=0.5, input_dropout=0.25,
                 max_iter=10000, verbose=0,
                 device=-1):

        self.skip_connection = skip_connection
        self.shared_embedding = shared_embedding
        self.shared_embedding_size = shared_embedding_size
        self.private_embedding_size = private_embedding_size
        self.activation = activation

        self.input_dropout = input_dropout
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = device
        self.lr = lr

    def fit(self, X, y, study_weights=None, callback=None):
        assert (self.shared_embedding in ['hard', 'adversarial',
                                          'hard+adversarial'])

        if self.optimizer == 'lbfgs':
            try:
                assert (self.dropout == 0)
                assert (self.input_dropout == 0)
                assert ('adversarial' not in self.shared_embedding)
            except AssertionError:
                raise ValueError('Dropout and adversarial training'
                                 'should not be used with LBFGS solver.')

        cuda, device = self._check_cuda()
        in_features = next(iter(X.values())).shape[1]
        n_samples = sum(len(this_X) for this_X in X.values())

        if study_weights is None:
            study_weights = {study: 1. for study in X}

        data_loaders = {}
        target_sizes = {}

        data_loader = DataLoader if self.optimizer == 'lbfgs' else \
            RepeatedDataLoader
        shuffle = self.optimizer != 'lbfgs'

        for study in X:
            target_sizes[study] = int(y[study]['contrast'].max()) + 1
            data_loaders[study] = data_loader(
                NiftiTargetDataset(X[study], y[study]),
                shuffle=shuffle,
                batch_size=self.batch_size, pin_memory=cuda)

        self.module_ = MultiTaskModule(
            in_features=in_features,
            activation=self.activation,
            shared_embedding=self.shared_embedding,
            shared_embedding_size=self.shared_embedding_size,
            private_embedding_size=self.private_embedding_size,
            skip_connection=self.skip_connection,
            input_dropout=self.input_dropout,
            dropout=self.dropout,
            target_sizes=target_sizes)
        self.loss_ = MultiTaskLoss(study_weights=study_weights,
                                   adversarial='adversarial'
                                               in self.shared_embedding)

        report_every = ceil(self.max_iter / self.verbose)

        if self.optimizer == 'lbfgs':
            def closure():
                self.module_.train()
                loss = 0
                for study, loader in data_loaders.items():
                    n_samples = 0
                    study_loss = 0
                    for input, target in loader:
                        target = target[:, [0, 2]]

                        if cuda:
                            input = input.cuda(device=device)
                            target = target.cuda(device=device)
                        input = Variable(input)
                        target = Variable(target)

                        n_samples += len(input)
                        input = {study: input}
                        target = {study: target}
                        pred = self.module_(input)

                        study_loss += self.loss_(target, pred)

                    loss += study_loss / n_samples
                return loss

            self.optimizer_ = LBFGSScipy(self.module_.parameters(),
                                         callback=callback,
                                         max_iter=self.max_iter,
                                         tolerance_grad=0,
                                         tolerance_change=0,
                                         report_every=report_every)
            self.optimizer_.step(closure)
            self.n_iter_ = self.optimizer_.n_iter_
        elif self.optimizer in ['adam', 'sgd']:
            data_loaders = {study: iter(loader) for study, loader in
                            data_loaders.items()}
            if self.optimizer == 'adam':
                self.optimizer_ = Adam(self.module_.parameters(), lr=self.lr,)
            else:
                self.optimizer_ = SGD(self.module_.parameters(), lr=self.lr,)
                self.scheduler_ = CosineAnnealingLR(self.optimizer_, T_max=5,
                                                    eta_min=self.lr * 1e-3)

            total_seen_samples = 0
            seen_samples = 0
            mean_loss = 0
            old_epoch = -1
            self.n_iter_ = 0
            self.module_.train()
            while self.n_iter_ < self.max_iter:
                if hasattr(self, 'scheduler_'):
                    self.scheduler_.step(self.n_iter_)
                self.optimizer_.zero_grad()
                for study, loader in data_loaders.items():
                    input, target = next(loader)
                    target = target[:, [0, 2]]

                    if cuda:
                        input = input.cuda(device=device)
                        target = target.cuda(device=device)
                    input = Variable(input)
                    target = Variable(target)

                    batch_size = len(input)
                    input = {study: input}
                    target = {study: target}
                    pred = self.module_(input)

                    this_loss = self.loss_(pred, target)
                    mean_loss += this_loss.data[0]
                    seen_samples += batch_size
                    total_seen_samples += batch_size

                    this_loss /= batch_size
                    this_loss.backward()

                    self.n_iter_ = total_seen_samples / n_samples
                self.optimizer_.step()

                epoch = floor(self.n_iter_)
                if report_every is not None and epoch > old_epoch \
                        and epoch % report_every == 0:
                    mean_loss = mean_loss / seen_samples
                    print('Epoch %.2f, train loss: % .4f' % (epoch, mean_loss))
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

    def predict_proba(self, X):
        cuda, device = self._check_cuda()

        data_loaders = {}
        for study, this_X in X.items():
            data_loaders[study] = DataLoader(NiftiTargetDataset(this_X),
                                             batch_size=len(this_X),
                                             shuffle=False,
                                             pin_memory=cuda)
        preds = {}
        study_preds = {}
        self.module_.eval()
        for study, loader in data_loaders.items():
            study_pred = []
            pred = []
            for (input, _) in loader:
                if cuda:
                    input = input.cuda(device=device)
                input = Variable(input, volatile=True)
                input = {study: input}
                this_study_pred, this_pred, _ = self.module_(input)[study]
                pred.append(this_pred)
                study_pred.append(this_study_pred)
            preds[study] = torch.cat(pred)
            study_preds[study] = torch.cat(study_pred)
        preds = {study: pred.data.cpu().numpy()
                 for study, pred in preds.items()}
        study_preds = {study: study_pred.data.cpu().numpy()
                       for study, study_pred in study_preds.items()}
        self.module_.train()
        return study_preds, preds

    def predict(self, X):
        study_preds, preds = self.predict_proba(X)
        study_preds = {study: np.argmax(study_pred, axis=1)
                       for study, study_pred in study_preds.items()}
        preds = {study: np.argmax(pred, axis=1)
                 for study, pred in preds.items()}
        dfs = {}
        for study in preds:
            pred = preds[study]
            study_pred = study_preds[study]
            dfs[study] = pd.DataFrame(
                dict(contrast=pred, study=study_pred, subject=0))
        return dfs

    @property
    def coef_(self):
        coefs = self.module_.coefs()
        return {study: coef.cpu().numpy() for study, coef in coefs.items()}

    @property
    def intercept_(self):
        intercepts = self.module_.intercepts()
        return {study: intercept.cpu().numpy()
                for study, intercept in intercepts.items()}

    @property
    def coef_cat_(self):
        return np.concatenate(list(self.coef_.values()), axis=1)

    @property
    def intercept_cat_(self):
        return np.concatenate(self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['module_', 'optimizer_', 'scheduler_']:
            if key in state:
                val = state.pop(key)
                with tempfile.SpooledTemporaryFile() as f:
                    torch.save(val, f)
                    f.seek(0)
                    state[key] = f.read()

        return state

    def __setstate__(self, state):
        disable_cuda = False
        for key in ['module_', 'optimizer_', 'scheduler_']:
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
