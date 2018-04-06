import tempfile
import warnings
from collections import defaultdict
from itertools import count
from math import ceil, floor
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from torch import nn
from torch.autograd import Variable, Function
from torch.nn import Linear, BatchNorm1d
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cogspaces.data import NiftiTargetDataset, infinite_iter
from cogspaces.model_selection import train_test_split
from cogspaces.optim.lbfgs import LBFGSScipy

from torch.nn.utils import vector_to_parameters, parameters_to_vector


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
                 shared_latent_size,
                 private_latent_size,
                 target_sizes,
                 input_dropout=0.,
                 dropout=0.,
                 decode=False,
                 adapt_size=0,
                 batch_norm=False,
                 activation='linear',
                 shared_latent='hard',
                 skip_connection=False):
        super().__init__()

        self.in_features = in_features
        self.shared_latent_size = shared_latent_size
        self.private_latent_size = private_latent_size

        self.decode = decode

        if activation == 'linear':
            self.activation = Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)
        self.input_dropout = nn.Dropout(p=input_dropout)

        self.skip_connection = skip_connection
        self.shared_latent = shared_latent

        self.adapt_size = adapt_size

        if self.adapt_size > 0:
            self.adapters = {}
            self.adapters = {study: Linear(in_features, adapt_size, bias=True)
                             for study in target_sizes}
            for study in target_sizes:
                self.add_module('adapters_%s' % study,
                                self.adapters[study])
            in_features = adapt_size

        if shared_latent_size > 0:
            def get_shared_embedder():
                return Linear(in_features, shared_latent_size, bias=True)

            def get_batch_norm():
                if batch_norm:
                    return BatchNorm1d(
                        num_features=(shared_latent_size + private_latent_size
                                      + in_features * skip_connection))
                else:
                    return Identity()

            if 'hard' in shared_latent:
                shared_embedder = get_shared_embedder()
                self.add_module('shared_embedder', shared_embedder)
                self.shared_embedders = {study: shared_embedder
                                         for study in target_sizes}
                # shared_batch_norm = get_batch_norm()
                # self.add_module('shared_batch_norm', shared_batch_norm)
                # self.shared_batch_norms = {study: shared_batch_norm
                #                           for study in target_sizes}
            else:
                self.shared_embedders = {study: get_shared_embedder()
                                         for study in target_sizes}
                for study in target_sizes:
                    self.add_module('shared_embedder_%s' % study,
                                    self.shared_embedders[study])
        self.shared_batch_norms = {study: get_batch_norm()
                                 for study in target_sizes}
        for study in target_sizes:
            self.add_module('shared_batch_norm_%s' % study,
                            self.shared_batch_norms[study])
        self.classifiers = {}
        if self.decode:
            self.decoders = {}

        if private_latent_size > 0:
            self.private_embedders = {}

        for study, size in target_sizes.items():
            if private_latent_size > 0:
                self.private_embedders[study] = \
                    Linear(in_features, private_latent_size, bias=True)
                self.add_module('private_embedder_%s' % study,
                                self.private_embedders[study])
                self.private_batch_norm = {
                    study: BatchNorm1d(num_features=in_features)
                    if batch_norm else Identity()
                    for study in target_sizes}
                for study in target_sizes:
                    self.add_module('private_batch_norm_%s' % study,
                                    self.private_batch_norm[study])

            self.classifiers[study] = \
                Linear(shared_latent_size + private_latent_size
                       + in_features * skip_connection, size)
            self.add_module('classifier_%s' % study, self.classifiers[study])

            if self.decode:
                self.decoders[study] = \
                    Linear(shared_latent_size + private_latent_size
                           + in_features * skip_connection, in_features)
                self.add_module('decoder_%s' % study, self.classifiers[study])

        if 'adversarial' in shared_latent:
            self.gradient_reversal = GradientReversal()
        if 'adversarial_study' in shared_latent:
            self.study_classifier = nn.Sequential(
                Linear(shared_latent_size, len(target_sizes), bias=True),
                nn.ReLU(),
                Linear(len(target_sizes), len(target_sizes), bias=True))
        if 'adversarial_contrast' in shared_latent:
            n_all_targets = sum(target_sizes.values())
            self.all_contrast_classifier = nn.Linear(shared_latent_size,
                                                     n_all_targets, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.ndimension() == 2:
                nn.init.xavier_uniform(param)
            elif param.ndimension() == 1:
                param.data.fill_(0.)

    def train(self, mode=True, freeze_batch_norm=False):
        super().train(mode)
        if freeze_batch_norm:
            for study, batch_norm in self.shared_batch_norms.items():
                batch_norm.eval()


    def get_latent(self, inputs):
        shared_latents = {}
        private_latents = {}
        for study, sub_input in inputs.items():
            latent = []
            if self.skip_connection:
                latent.append(sub_input)

            sub_input = self.input_dropout(sub_input)

            if self.adapt_size > 0:
                sub_input = self.activation(self.adapters[study](sub_input))

            if self.shared_latent_size > 0:
                shared_latents[study] = self.dropout(
                    self.shared_batch_norms[study](self.activation(
                        self.shared_embedders[study](sub_input))))
            else:
                shared_latents[study] = None

            if self.private_latent_size > 0:
                private_latents[study] = self.dropout(
                    self.private_batch_norms[study](self.activation(
                        self.private_embedders[study](sub_input))))
            else:
                private_latents[study] = None

        return shared_latents, private_latents

    def forward(self, inputs):
        shared_latents, private_latents = self.get_latent(inputs)

        preds = {}

        for study in shared_latents:
            latent_list = []
            shared_latent = shared_latents[study]
            if shared_latent is not None:
                latent_list.append(shared_latent)
            private_latent = private_latents[study]
            if private_latent is not None:
                latent_list.append(shared_latent)
            latent = torch.cat(latent_list, dim=1)

            pred_contrast = F.log_softmax(self.classifiers[study](latent),
                                          dim=1)

            if 'adversarial_study' in self.shared_latent:
                pred_study = F.log_softmax(self.study_classifier(
                    self.gradient_reversal(shared_latent)), dim=1)
            else:
                pred_study = None
            if 'adversarial_contrast' in self.shared_latent:
                pred_all_contrast = F.log_softmax(
                    self.all_contrast_classifier(
                        self.gradient_reversal(shared_latent)), dim=1)
            else:
                pred_all_contrast = None
            if (self.shared_latent_size > 0 and
                    self.private_latent_size > 0 is not None):
                corr = (torch.matmul(
                    shared_latent.transpose(0, 1), private_latent) /
                        self.private_latent_size /
                        self.shared_latent_size)
                penalty = torch.sum(torch.abs(corr))
            else:
                penalty = None
            if self.decode:
                decoded = self.decoders[study](latent)
            else:
                decoded = None
            preds[study] = (pred_study, pred_contrast, pred_all_contrast,
                            penalty, decoded)
        return preds

    def latent(self):
        return {study: shared_embedder.weight.data for study, shared_embedder
                in self.shared_embedders.items()}


    def coefs(self):
        coefs = {}
        for study in self.classifiers:
            coef = self.classifiers[study].weight.data
            if self.skip_connection:
                skip_coef = coef[:, :self.in_features]
                if self.in_features < coef.shape[1]:
                    coef = coef[:, self.in_features:]
            else:
                skip_coef = 0
            if self.shared_latent_size > 0:
                shared_coef = coef[:, :self.shared_latent_size]
                shared_embed = self.shared_embedders[study].weight.data
                shared_coef = torch.matmul(shared_coef, shared_embed)
            else:
                shared_coef = 0
            if self.private_latent_size > 0:
                private_coef = coef[:, -self.private_latent_size:]
                private_embed = self.private_embedders[study].weight.data
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
                targets: Dict[str, torch.LongTensor]) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor]:
        loss = 0
        true_loss = 0
        for study in inputs:
            study_pred, pred, all_contrast_pred, penalty, \
            decoded = inputs[study]
            study_target, target, all_contrast_target, data = targets[study]

            this_loss = (F.nll_loss(pred, target, size_average=True)
                         * self.loss_weights['contrast'])
            if decoded is not None:
                decoding_loss = F.mse_loss(decoded, data, size_average=True)
                this_loss += decoding_loss * self.loss_weights['decoding']

            if penalty is not None:
                penalty /= pred.shape[0]
                this_loss += penalty * self.loss_weights['penalty']
            this_true_loss = this_loss
            if study_pred is not None:
                study_loss = F.nll_loss(study_pred, study_target,
                                        size_average=True)
                this_true_loss = (this_true_loss - study_loss *
                                  self.loss_weights['study'])
                this_loss += study_loss * self.loss_weights['study']

            if all_contrast_pred is not None:
                all_contrast = F.nll_loss(all_contrast_pred,
                                          all_contrast_target,
                                          size_average=True)
                this_true_loss = (this_true_loss - all_contrast *
                                  self.loss_weights['all_contrast'])
                this_loss += all_contrast * self.loss_weights['all_contrast']

            loss += this_loss * self.study_weights[study]
            true_loss += this_true_loss * self.study_weights[study]
        return loss, true_loss.detach()


def sampler(dictionary, seed=None):
    keys, values = zip(*list(dictionary.items()))
    values = np.array(values)
    values /= np.sum(values)
    random_state = check_random_state(seed)
    while True:
        res = random_state.choice(keys, p=values)
        yield res

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def next_batches(data_loaders, cuda, device, sampling='all',
                 study_weights=None, seed=None):
    if sampling == 'all':
        inputs = {}
        targets = {}
        batch_sizes = 0
        loaders_iter = data_loaders.keys()
        outer = count(0, 1)
    else:
        outer = range(0, 1)
        if sampling == 'cycle':
            loaders_iter = infinite_iter(data_loaders.keys())
        elif sampling == 'weighted_random':
            loaders_iter = sampler(study_weights, seed=seed)
        else:
            raise ValueError
    for _ in outer:
        inputs = {}
        targets = {}
        batch_sizes = 0
        for study in loaders_iter:
            loader = data_loaders[study]
            input, study_target, target, target_all_contrast = next(loader)
            batch_size = input.shape[0]
            if cuda:
                input = input.cuda(device=device)
                target = target.cuda(device=device)
                target_all_contrast = target_all_contrast.cuda(device=device)
                study_target = study_target.cuda(device=device)
            input = Variable(input)
            target = Variable(target)
            study_target = Variable(study_target)
            target_all_contrast = Variable(target_all_contrast)
            if sampling == 'all':
                inputs[study] = input
                targets[
                    study] = study_target, target, target_all_contrast, input
                batch_sizes += batch_size
            else:
                yield {study: input}, {
                    study: (study_target, target, target_all_contrast,
                            input)}, batch_size
        if sampling == 'all':
            yield inputs, targets, batch_sizes


class EnsembleFactoredClassifier(BaseEstimator):
    def __init__(self,
                 skip_connection=False,
                 shared_latent_size=30,
                 private_latent_size=5,
                 activation='linear',
                 shared_latent='hard',
                 batch_size=128, optimizer='sgd',
                 decode=False,
                 loss_weights=None,
                 epoch_counting='all',
                 lr=0.001,
                 fine_tune=False,
                 dropout=0.5, input_dropout=0.25,
                 max_iter=10000, verbose=0,
                 device=-1,
                 sampling='cycle',
                 averaging=False,
                 seed=None,
                 n_jobs=1,
                 n_runs=1):
        self.skip_connection = skip_connection
        self.shared_latent = shared_latent
        self.shared_latent_size = shared_latent_size
        self.private_latent_size = private_latent_size
        self.activation = activation
        self.decode = decode
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = device
        self.lr = lr
        self.sampling = sampling
        self.fine_tune = fine_tune
        self.loss_weights = loss_weights
        self.seed = seed

        self.epoch_counting = epoch_counting

        self.averaging = averaging

        self.n_jobs = n_jobs
        self.n_runs = n_runs

    def fit(self, X, y, study_weights, callback=None):
        estimator = FactoredClassifier(
            skip_connection=self.skip_connection,
            shared_latent_size=self.shared_latent_size,
            private_latent_size=self.private_latent_size,
            activation=self.activation,
            shared_latent=self.shared_latent,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            decode=self.decode,
            loss_weights=self.loss_weights,
            lr=self.lr,
            epoch_counting='target',
            averaging=self.averaging,
            fine_tune=self.fine_tune,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            max_iter=self.max_iter,
            verbose=self.verbose,
            device=self.device,
            sampling=self.sampling,
            seed=self.seed)

        seeds = check_random_state(self.seed).randint(0, np.iinfo('int32').max,
                                                      size=self.n_runs)
        self.estimators = Parallel(n_jobs=self.n_jobs)(delayed(_fit)(
            clone(estimator), X, y, study_weights, callback, int(seed))
                                                       for seed in seeds)
        return self

    def predict_proba(self, X):
        mean_preds = defaultdict(lambda: 0)
        mean_study_preds = defaultdict(lambda: 0)
        mean_all_contrast_preds = defaultdict(lambda: 0)
        for estimator in self.estimators:
            study_preds, preds, all_contrast_preds = estimator.predict_proba(X)
            for study in X:
                mean_preds[study] += preds[study]
                if study_preds is not None:
                    mean_study_preds[study] += study_preds[study]
                if all_contrast_preds is not None:
                    mean_all_contrast_preds[study] += all_contrast_preds[study]
        for study in X:
            mean_preds[study] /= len(self.estimators)
            if study_preds is not None:
                mean_study_preds[study] /= len(self.estimators)
            if all_contrast_preds is not None:
                mean_all_contrast_preds[study] /= len(self.estimators)
        if not study_preds:
            study_preds = None
        if not mean_all_contrast_preds:
            all_contrast_preds = None
        return study_preds, preds, all_contrast_preds

    def predict(self, X):
        # Totally scandalous
        return FactoredClassifier.predict(self, X)

    @property
    def coef_(self):
        return self.estimator.coef_

    @property
    def intercept_(self):
        return self.estimator.intercept_


def _fit(estimator, X, y, study_weights, callback, seed):
    estimator.seed = seed
    estimator.fit(X, y, study_weights, callback)
    return estimator


class FactoredClassifier(BaseEstimator):
    def __init__(self,
                 skip_connection=False,
                 adapt_size=0,
                 shared_latent_size=30,
                 private_latent_size=5,
                 activation='linear',
                 shared_latent='hard',
                 batch_size=128, optimizer='sgd',
                 decode=False,
                 loss_weights=None,
                 epoch_counting='all',
                 lr=0.001,
                 fine_tune=False,
                 dropout=0.5, input_dropout=0.25,
                 max_iter=10000, verbose=0,
                 device=-1,
                 sampling='cycle',
                 batch_norm=False,
                 averaging=False,
                 patience=100,
                 seed=None):

        self.skip_connection = skip_connection
        self.shared_latent = shared_latent
        self.shared_latent_size = shared_latent_size
        self.private_latent_size = private_latent_size
        self.activation = activation
        self.decode = decode

        self.input_dropout = input_dropout
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.epoch_counting = epoch_counting
        self.verbose = verbose
        self.device = device
        self.lr = lr
        self.sampling = sampling

        self.adapt_size = adapt_size

        self.averaging = averaging

        self.fine_tune = fine_tune

        self.loss_weights = loss_weights

        self.seed = seed

        self.batch_norm = batch_norm

        self.patience = patience

    def fit(self, X, y, study_weights=None, callback=None):
        cuda, device = self._check_cuda()
        in_features = next(iter(X.values())).shape[1]
        if self.epoch_counting == 'all':
            n_samples = sum(len(this_X) for this_X in X.values())
        elif self.epoch_counting == 'target':
            if self.sampling == 'weighted_random':
                multiplier = (sum(study_weights.values()) /
                              next(iter(study_weights.values())))
            else:
                multiplier = len(X)
            print(multiplier)
            n_samples = len(next(iter(X.values()))) * multiplier
        else:
            raise ValueError

        if study_weights is None:
            study_weights = {study: 1. for study in X}
        if self.loss_weights is None:
            loss_weights = {'contrast': 1, 'study': 1, 'penalty': 1,
                            'decoding': 1, 'all_contrast': 1}
        else:
            loss_weights = self.loss_weights

        data_loaders = {}
        target_sizes = {}

        torch.manual_seed(self.seed)

        for study in X:
            target_sizes[study] = int(y[study]['contrast'].max()) + 1

        if self.shared_latent_size == 'auto':
            shared_latent_size = sum(target_sizes.values())
        else:
            shared_latent_size = self.shared_latent_size

        self.module_ = MultiTaskModule(
            in_features=in_features,
            activation=self.activation,
            decode=self.decode,
            batch_norm=self.batch_norm,
            adapt_size=self.adapt_size,
            shared_latent=self.shared_latent,
            shared_latent_size=shared_latent_size,
            private_latent_size=self.private_latent_size,
            skip_connection=self.skip_connection,
            input_dropout=self.input_dropout,
            dropout=self.dropout,
            target_sizes=target_sizes)

        if self.sampling == 'weighted_random':
            loss_study_weights = {study: 1. for study in X}
        else:
            loss_study_weights = study_weights

        self.loss_ = MultiTaskLoss(
            study_weights=loss_study_weights,
            loss_weights=loss_weights)

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
                loss, true_loss = self.loss_(preds_, targets_)
                return loss

            optimizer = LBFGSScipy(self.module_.parameters(),
                                   callback=callback,
                                   max_iter=self.max_iter,
                                   tolerance_grad=0,
                                   tolerance_change=0,
                                   report_every=2)
            optimizer.step(closure)
        else:
            self.module_.reset_parameters()
            for study in X:
                data_loaders[study] = DataLoader(
                    NiftiTargetDataset(X[study], y[study]),
                    shuffle=True,
                    batch_size=self.batch_size, pin_memory=cuda)

            self.n_iter_ = 0
            old_epoch = -1
            seen_samples = 0
            if self.fine_tune:
                phases = ['joined', 'fine_tune']
            else:
                phases = ['joined']

            params = self.module_.parameters()
            if self.optimizer == 'adam':
                optimizer = Adam(params, lr=self.lr)
                scheduler = None
            elif self.optimizer == 'sgd':
                optimizer = SGD(params, lr=self.lr,)
                scheduler = CosineAnnealingLR(optimizer, T_max=30,
                                              eta_min=1e-3 * self.lr)
            else:
                raise ValueError

            for phase in phases:
                if phase == 'joined':
                    max_iter = self.max_iter
                else:
                    self.module_.input_dropout.p = 0.
                    # self.module_.dropout.p = 0.75
                    for study in X:
                        self.module_.shared_embedders[study].weight.requires_grad = False
                        self.module_.shared_embedders[study].bias.requires_grad = False
                    optimizer.param_groups[0]['lr'] *= .1
                    # d = weight.shape[0]
                    # rot = weight.new(d, d)
                    # nn.init.orthogonal(rot)
                    # shared_embedder.weight.data = rot @ weight
                    # self.module_.dropout.p = 0.
                    # self.module_.input_dropout.p = 0.
                    max_iter = old_epoch + self.max_iter

                infinite_data_loaders = {study: infinite_iter(loader) for
                                          study, loader in
                                          data_loaders.items()}
                # Logging logic
                best_loss = float('inf')
                epoch_loss = 0
                epoch_seen_samples = 0
                no_improvement = 0
                if self.verbose != 0:
                    report_every = ceil(self.max_iter / self.verbose)
                else:
                    report_every = None
                best_params = parameters_to_vector(
                    self.module_.parameters())
                best_params_list = [best_params]
                for inputs, targets, batch_size in next_batches(
                        infinite_data_loaders, cuda=cuda, device=device,
                        sampling=self.sampling,
                        study_weights=study_weights,
                        seed=self.seed):
                    if scheduler is not None:
                        scheduler.step(self.n_iter_)
                    self.module_.train(
                        # freeze_batch_norm=phase == 'fine_tune'
                    )
                    optimizer.zero_grad()
                    preds = self.module_(inputs)
                    this_loss, this_true_loss = self.loss_(preds, targets)
                    this_loss.backward()
                    optimizer.step()
                    seen_samples += batch_size
                    epoch_seen_samples += batch_size
                    epoch_loss += this_true_loss.data[0] * batch_size
                    self.n_iter_ = seen_samples / n_samples
                    epoch = floor(self.n_iter_)
                    if epoch > old_epoch:
                        old_epoch = epoch
                        epoch_loss /= epoch_seen_samples
                        epoch_seen_samples = 0
                        if epoch_loss > best_loss:
                            no_improvement += 1
                        else:
                            no_improvement = 0
                            best_loss = epoch_loss
                            best_params = parameters_to_vector(
                                self.module_.parameters())
                            if self.averaging:
                                if len(best_params_list) >= 5:
                                    best_params_list = best_params_list[1:]
                                best_params_list.append(best_params)
                        if (report_every is not None
                                and epoch % report_every == 0
                                or no_improvement > self.patience
                                or epoch >= max_iter):
                            print('Epoch %.2f, train loss: %.4f'
                                  % (epoch, epoch_loss))
                            if callback is not None:
                                callback(self, self.n_iter_)
                        if no_improvement > self.patience:
                            print('Stopping at epoch %.2f, best train loss'
                                  ' %.4f' % (epoch, best_loss))
                            if phase == 'joined' and len(phases) == 2:
                                print('Fine tuning...')
                            break
                        if epoch >= max_iter:
                            print('Hard stopping at epoch %.2f,'
                                  ' best train loss %.4f'
                                  % (epoch, best_loss))
                            if phase == 'joined' and len(phases) == 2:
                                print('Fine tuning...')
                            break
        if 'adversarial' not in self.shared_latent:
            if self.averaging:
                best_params = torch.cat([params[None, :]
                                         for params in best_params_list],
                                        dim=0)
                best_params = torch.mean(best_params, dim=0)
                vector_to_parameters(best_params, self.module_.parameters())
        return self

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

    def predict_latent(self, X):
        cuda, device = self._check_cuda()
        data_loaders = {}
        for study, this_X in X.items():
            data_loaders[study] = DataLoader(NiftiTargetDataset(this_X),
                                             batch_size=len(this_X),
                                             shuffle=False,
                                             pin_memory=cuda)
        self.module_.eval()
        latents = {}
        for study, loader in data_loaders.items():
            latent = []
            for (input, _, _, _) in loader:
                if cuda:
                    input = input.cuda(device=device)
                input = Variable(input, volatile=True)
                input = {study: input}
                this_latent, _ = self.module_.get_latent(input)
                latent.append(this_latent[study])
            latents[study] = torch.cat(latent, dim=0).data.cpu().numpy()
        return latents

    def predict_proba(self, X):
        cuda, device = self._check_cuda()

        data_loaders = {}
        for study, this_X in X.items():
            data_loaders[study] = DataLoader(NiftiTargetDataset(this_X),
                                             batch_size=len(this_X),
                                             shuffle=False,
                                             pin_memory=cuda)
        preds = {}
        if 'adversarial_study' in self.shared_latent:
            study_preds = {}
        else:
            study_preds = None
        if 'adversarial_contrast' in self.shared_latent:
            all_contrast_preds = {}
        else:
            all_contrast_preds = None
        self.module_.eval()
        for study, loader in data_loaders.items():
            if study_preds is not None:
                study_pred = []
            if all_contrast_preds is not None:
                all_contrast_pred = []
            pred = []
            for (input, _, _, _) in loader:
                if cuda:
                    input = input.cuda(device=device)
                input = Variable(input, volatile=True)
                input = {study: input}
                this_study_pred, this_pred, this_all_contrast_pred, \
                _, _ = self.module_(input)[study]
                pred.append(this_pred)
                if study_preds is not None:
                    study_pred.append(this_study_pred)
                if all_contrast_preds is not None:
                    all_contrast_pred.append(this_all_contrast_pred)
            preds[study] = torch.cat(pred)
            if study_preds is not None:
                study_preds[study] = torch.cat(study_pred)
            if all_contrast_preds is not None:
                all_contrast_preds[study] = torch.cat(all_contrast_pred)
        preds = {study: pred.data.cpu().numpy()
                 for study, pred in preds.items()}
        if study_preds is not None:
            study_preds = {study: study_pred.data.cpu().numpy()
                           for study, study_pred in study_preds.items()}
        if all_contrast_preds is not None:
            all_contrast_preds = {study: all_contrast_pred.data.cpu().numpy()
                                  for study, all_contrast_pred in
                                  all_contrast_preds.items()}
        return study_preds, preds, all_contrast_preds

    def predict(self, X):
        study_preds, preds, all_contrast_preds = self.predict_proba(X)
        if study_preds is not None:
            study_preds = {study: np.argmax(study_pred, axis=1)
                           for study, study_pred in study_preds.items()}
        else:
            study_preds = {study: 0 for study in preds}
        if all_contrast_preds is not None:
            for study, all_contrast_pred in all_contrast_preds.items():
                all_contrast_preds[study] = np.argmax(all_contrast_pred,
                                                      axis=1)
        else:
            all_contrast_preds = {study: 0 for study in preds}
        preds = {study: np.argmax(pred, axis=1)
                 for study, pred in preds.items()}
        dfs = {}
        for study in preds:
            pred = preds[study]
            study_pred = study_preds[study]
            all_contrast_pred = all_contrast_preds[study]
            dfs[study] = pd.DataFrame(
                dict(contrast=pred, study=study_pred,
                     all_contrast=all_contrast_pred,
                     subject=0))
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


class FactoredClassifierCV(BaseEstimator):
    def __init__(self,
                 skip_connection=False,
                 shared_latent_size=30,
                 private_latent_size=5,
                 activation='linear',
                 shared_latent='hard',
                 batch_size=128, optimizer='sgd',
                 decode=False,
                 loss_weights=None,
                 lr=0.001,
                 fine_tune=False,
                 dropout=0.5, input_dropout=0.25,
                 max_iter=10000, verbose=0,
                 device=-1,
                 sampling='cycle',
                 n_jobs=1,
                 averaging=False,
                 n_splits=3,
                 n_runs=10,
                 seed=None):
        self.skip_connection = skip_connection
        self.shared_latent = shared_latent
        self.shared_latent_size = shared_latent_size
        self.private_latent_size = private_latent_size
        self.activation = activation
        self.decode = decode
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = device
        self.lr = lr
        self.sampling = sampling
        self.fine_tune = fine_tune
        self.loss_weights = loss_weights
        self.seed = seed

        self.averaging = averaging

        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.n_runs = n_runs

    def fit(self, X, y, study_weights, callback=None):
        studies = list(X.keys())
        target, sources = studies[0], studies[1:]
        print(study_weights)

        classifier = EnsembleFactoredClassifier(
            skip_connection=self.skip_connection,
            shared_latent_size=self.shared_latent_size,
            private_latent_size=self.private_latent_size,
            activation=self.activation,
            shared_latent=self.shared_latent,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            decode=self.decode,
            loss_weights=self.loss_weights,
            lr=self.lr,
            n_runs=1,
            epoch_counting='target',
            averaging=self.averaging,
            fine_tune=self.fine_tune,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            max_iter=self.max_iter,
            verbose=self.verbose,
            device=self.device,
            sampling=self.sampling,
            seed=self.seed)

        if len(sources) > 0:
            seeds = check_random_state(self.seed).randint(0,
                                                          np.iinfo(
                                                              'int32').max,
                                                          size=self.n_splits)
            rolled = Parallel(n_jobs=self.n_jobs)(
                delayed(eval_transferability)(
                    classifier, X, y, target, source, seed,
                    study_weights)
                for seed in seeds
                for source in [None] + sources)
            transfer = []
            rolled = iter(rolled)
            for seed in seeds:
                baseline_score = next(rolled)
                for source in sources:
                    score = next(rolled)
                    transfer.append(dict(seed=seed, source=source,
                                         score=score,
                                         diff=score - baseline_score))
            transfer = pd.DataFrame(transfer)
            transfer = transfer.set_index(['source', 'seed'])
            transfer = transfer.groupby('source').agg('mean')
            print('Pair transfers:')
            transfer = transfer.sort_values('diff', ascending=False)
            print(transfer)
            transfer = transfer.query('diff > 0.005')
            positive = transfer.index.get_level_values(
                'source').values.tolist()
            weights = transfer['diff'] / transfer['diff'].sum()
            weights = weights.to_dict()
            print(weights)
            print('Transfering datasets:', positive)
            self.studies_ = [target] + positive
            if len(positive) > 1:
                study_weights = {study: study_weights[study] * weights[study]
                                        / study_weights[target]
                                 for study in positive}
            study_weights[target] = 1
            print('New weights:', study_weights)
        else:
            self.studies_ = [target]
        X = {study: X[study] for study in self.studies_}
        y = {study: y[study] for study in self.studies_}
        classifier.set_params(max_iter=self.max_iter, n_runs=self.n_runs,
                              n_jobs=self.n_jobs)
        self.classifier_ = classifier.fit(X, y, study_weights=study_weights)
        return self

    def predict(self, X):
        return self.classifier_.predict(X)


def eval_transferability(classifier, X, y, target, source=None, seed=0,
                         study_weights=None,
                         callback=None):
    train_data, val_data, train_targets, val_targets = \
        train_test_split(X, y, random_state=seed)
    y_val_true = val_targets[target]['contrast'].values
    X_val = {target: val_data[target]}
    if source is None:
        train_data = {target: train_data[target]}
        train_targets = {target: train_targets[target]}
        study_weights = {target: 1}
    else:
        train_data = {target: train_data[target],
                      source: train_data[source]}
        train_targets = {target: train_targets[target],
                         source: train_targets[source]}
        if study_weights is not None:
            study_weights = {target: 1,
                             source: study_weights[source]
                                     / study_weights[target]}
    classifier.fit(train_data, train_targets,
                   study_weights=study_weights, callback=callback)
    y_val_pred = classifier.predict(X_val)[target]['contrast'].values
    return accuracy_score(y_val_true, y_val_pred)
