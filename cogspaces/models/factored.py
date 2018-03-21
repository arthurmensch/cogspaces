import tempfile
import warnings
from math import ceil, floor, sqrt
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from torch import nn
from torch.autograd import Variable, Function
from torch.nn import Linear
from torch.nn.functional import nll_loss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cogspaces.data import NiftiTargetDataset, infinite_iter
from cogspaces.optim.lbfgs import LBFGSScipy
import torch.nn.functional as F

class GradReverseFunc(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class GradientReversal(nn.Module):
    def forward(self, input):
        return GradReverseFunc.apply(input)


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
                return Linear(in_features, shared_embedding_size, bias=False)

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
                self.private_embedders[study] = \
                    Linear(in_features, private_embedding_size, bias=False),
                self.add_module('private_embedder_%s' % study,
                                self.private_embedders[study])

            self.classifiers[study] =\
                Linear(shared_embedding_size + private_embedding_size
                       + in_features * skip_connection, size)
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
                    study_pred = self.study_classifier(
                        self.gradient_reversal(shared_embedding))
                else:
                    study_pred = None
                embedding.append(shared_embedding)
            else:
                study_pred = None
                shared_embedding = None

            if self.private_embedding_size > 0:
                private_embedding = self.private_embedders[study](sub_input)
                embedding.append(private_embedding)
            else:
                private_embedding = None

            if shared_embedding is not None and private_embedding is not None:
                corr = torch.matmul(shared_embedding.transpose(0, 1),
                                    private_embedding)
                corr /= self.private_embedding_size *\
                        self.shared_embedding_size
                penalty = torch.sum(torch.abs(corr))
            else:
                penalty = 0

            if len(embedding) > 1:
                embedding = torch.cat(embedding, dim=1)
            else:
                embedding = embedding[0]
            pred = F.log_softmax(self.classifiers[study](
                self.dropout(self.activation(embedding))), dim=1)

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
    def __init__(self, study_weights: Dict[str, float],
                 loss_weights: Dict[str, float]) -> None:
        super().__init__()
        self.loss_weights = loss_weights
        self.study_weights = study_weights

    def forward(self, inputs: Dict[str, torch.FloatTensor],
                targets: Dict[str, torch.LongTensor]) -> torch.FloatTensor:
        loss = 0
        for study in inputs:
            study_pred, pred, penalty = inputs[study]
            study_target, target = targets[study][0], targets[study][1]

            this_loss = (nll_loss(pred, target, size_average=True)
                         * self.loss_weights['contrast'])

            if penalty != 0:
                penalty /= pred.shape[0]
                this_loss += penalty * self.loss_weights['penalty']

            if study_pred is not None:
                study_loss = nll_loss(study_pred, study_target,
                                      size_average=True)
                this_loss += study_loss * self.loss_weights['adversarial']

            loss += this_loss * self.study_weights[study]
        return loss


def next_batches(data_loaders, cuda, device, cycle=True):
    if not cycle:
        inputs = {}
        targets = {}
        batch_sizes = 0
    for study, loader in data_loaders.items():
        input, study_target, target = next(loader)
        batch_size = input.shape[0]
        if cuda:
            input = input.cuda(device=device)
            target = target.cuda(device=device)
            study_target = study_target.cuda(device=device)
        input = Variable(input)
        target = Variable(target)
        study_target = Variable(study_target)
        if cycle:
            yield {study: input}, {study: [study_target, target]}, batch_size
        else:
            inputs[study] = input
            targets[study] = [study_target, target]
            batch_sizes += batch_size
    if not cycle:
        yield inputs, targets, batch_sizes


class FactoredClassifier(BaseEstimator):
    def __init__(self,
                 skip_connection=False,
                 shared_embedding_size=30,
                 private_embedding_size=5,
                 activation='linear',
                 shared_embedding='hard+adversarial',
                 batch_size=128, optimizer='sgd',
                 loss_weights=None,
                 lr=0.001,
                 fine_tune=False,
                 dropout=0.5, input_dropout=0.25,
                 max_iter=10000, verbose=0,
                 device=-1,
                 cycle=True,
                 seed=None):

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
        self.cycle = cycle

        self.fine_tune = fine_tune

        self.loss_weights = loss_weights

        self.seed = seed

    def fit(self, X, y, study_weights=None, callback=None):
        assert (self.shared_embedding in ['hard', 'adversarial',
                                          'hard+adversarial'])

        cuda, device = self._check_cuda()
        in_features = next(iter(X.values())).shape[1]
        n_samples = sum(len(this_X) for this_X in X.values())

        if study_weights is None:
            study_weights = {study: 1. for study in X}

        data_loaders = {}
        target_sizes = {}

        torch.manual_seed(self.seed)

        for study in X:
            target_sizes[study] = int(y[study]['contrast'].max()) + 1

        if self.shared_embedding_size == 'auto':
            shared_embedding_size = sum(target_sizes.values())
        else:
            shared_embedding_size = self.shared_embedding_size

        self.module_ = MultiTaskModule(
            in_features=in_features,
            activation=self.activation,
            shared_embedding=self.shared_embedding,
            shared_embedding_size=shared_embedding_size,
            private_embedding_size=self.private_embedding_size,
            skip_connection=self.skip_connection,
            input_dropout=self.input_dropout,
            dropout=self.dropout,
            target_sizes=target_sizes)
        self.loss_ = MultiTaskLoss(study_weights=study_weights,
                                   loss_weights=self.loss_weights)

        if self.optimizer == 'lbfgs':
            for study in X:
                data_loaders[study] = DataLoader(
                    NiftiTargetDataset(X[study], y[study]),
                    len(X[study]), shuffle=False, pin_memory=cuda)

            # Desactivate dropout
            def closure():
                data_loaders_iter = {study: iter(data_loader_)
                                     for study, data_loader_
                                     in data_loaders.items()}
                inputs_, targets_, _ = next_batches(data_loaders_iter,
                                                    cuda=cuda,
                                                    device=device)
                self.module_.eval()
                preds_ = self.module_(inputs_)
                return self.loss_(preds_, targets_)

            optimizer = LBFGSScipy(self.module_.parameters(),
                                   callback=callback,
                                   max_iter=self.max_iter,
                                   tolerance_grad=0,
                                   tolerance_change=0,
                                   report_every=2)
            optimizer.step(closure)
        else:
            for study in X:
                data_loaders[study] = DataLoader(
                    NiftiTargetDataset(X[study], y[study]),
                    shuffle=True,
                    batch_size=self.batch_size, pin_memory=cuda)

            data_loaders = {study: infinite_iter(loader) for study, loader in
                            data_loaders.items()}

            if self.optimizer == 'adam':
                self.optimizer_ = Adam(self.module_.parameters(), lr=self.lr, )
                self.scheduler_ = None
            else:
                self.optimizer_ = SGD(self.module_.parameters(), lr=self.lr, )
                self.scheduler_ = CosineAnnealingLR(self.optimizer_, T_max=10,
                                                    eta_min=1e-3 * self.lr)

            self.n_iter_ = 0
            # Logging logic
            old_epoch = -1
            seen_samples = 0
            epoch_loss = 0
            epoch_seen_samples = 0
            report_every = ceil(self.max_iter / self.verbose)

            while self.n_iter_ < self.max_iter:
                if self.scheduler_ is not None:
                    self.scheduler_.step(self.n_iter_)
                for inputs, targets, batch_size in next_batches(data_loaders,
                                                                cuda=cuda,
                                                                device=device,
                                                                cycle=self.cycle):
                    self.module_.train()
                    self.optimizer_.zero_grad()
                    preds = self.module_(inputs)
                    this_loss = self.loss_(preds, targets)
                    this_loss.backward()
                    self.optimizer_.step()
                    seen_samples += batch_size
                    epoch_seen_samples += batch_size
                    epoch_loss += this_loss * batch_size
                self.n_iter_ = seen_samples / n_samples
                epoch = floor(self.n_iter_)
                if report_every is not None and epoch > old_epoch \
                        and epoch % report_every == 0:
                    epoch_loss /= epoch_seen_samples
                    print('Epoch %.2f, train loss: % .4f'
                          % (epoch, epoch_loss))
                    epoch_loss = 0
                    epoch_seen_samples = 0
                    if callback is not None:
                        callback(self.n_iter_)
                old_epoch = epoch

        if self.fine_tune:
            print('Fine tuning')
            parameters = []
            for classifier in self.module_.classifiers.values():
                parameters += list(classifier.parameters())
            if 'adversarial' in self.shared_embedding:
                parameters += list(self.module_.study_classifier.parameters())

            for study in X:
                data_loaders[study] = DataLoader(
                    NiftiTargetDataset(X[study], y[study]),
                    len(X[study]), shuffle=False, pin_memory=cuda)

            # Desactivate dropout
            def closure():
                data_loaders_iter = {study: iter(data_loader_) for study,
                                                                   data_loader_
                                     in data_loaders.items()}
                inputs_, targets_, _ = next_batches(data_loaders_iter,
                                                    cuda=cuda,
                                                    device=device)
                self.module_.eval()
                preds_ = self.module_(inputs_)
                return self.loss_(preds_, targets_)

            optimizer = LBFGSScipy(parameters,
                                   callback=callback,
                                   max_iter=self.max_iter,
                                   tolerance_grad=0,
                                   tolerance_change=0,
                                   report_every=2)
            optimizer.step(closure)

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
        if 'adversarial' in self.shared_embedding:
            study_preds = {}
        else:
            study_preds = None
        self.module_.eval()
        for study, loader in data_loaders.items():
            if study_preds is not None:
                study_pred = []
            pred = []
            for (input, _, _) in loader:
                if cuda:
                    input = input.cuda(device=device)
                input = Variable(input, volatile=True)
                input = {study: input}
                this_study_pred, this_pred, _ = self.module_(input)[study]
                pred.append(this_pred)
                if study_preds is not None:
                    study_pred.append(this_study_pred)
            preds[study] = torch.cat(pred)
            if study_preds is not None:
                study_preds[study] = torch.cat(study_pred)
        preds = {study: pred.data.cpu().numpy()
                 for study, pred in preds.items()}
        if study_preds is not None:
            study_preds = {study: study_pred.data.cpu().numpy()
                           for study, study_pred in study_preds.items()}
        return study_preds, preds

    def predict(self, X):
        study_preds, preds = self.predict_proba(X)
        if study_preds is not None:
            study_preds = {study: np.argmax(study_pred, axis=1)
                           for study, study_pred in study_preds.items()}
        else:
            study_preds = {study: 0 for study in preds}
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
