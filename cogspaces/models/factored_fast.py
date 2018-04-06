import itertools
import tempfile
import warnings
from math import ceil, floor
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

from cogspaces.models.factored import Identity
from cogspaces.optim.lbfgs import LBFGSScipy


def infinite_iter(iterable):
    while True:
        for elem in iterable:
            yield elem


class Embedder(nn.Module):
    def __init__(self, in_features, latent_size, input_dropout=0.,
                 activation='linear'):
        super().__init__()

        self.input_dropout = nn.Dropout(p=input_dropout)

        self.linear = nn.Linear(in_features, latent_size, bias=True)

        if activation == 'linear':
            self.activation = Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, input):
        return self.activation(self.linear(self.input_dropout(input)))


class LatentClassifier(nn.Module):
    def __init__(self, latent_size, target_size, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(latent_size)
        self.linear = nn.Linear(latent_size, target_size, bias=True)

    def forward(self, input):
        return F.log_softmax(
            self.linear(self.dropout(self.batch_norm(input))), dim=1)


class MultiStudyModule(nn.Module):
    def __init__(self, in_features,
                 latent_size,
                 target_sizes,
                 input_dropout=0.,
                 dropout=0.,
                 activation='linear'):
        super().__init__()

        self.embedder = Embedder(in_features, latent_size, input_dropout,
                                 activation)

        self.classifiers = {study: LatentClassifier(latent_size, target_size,
                                                    dropout)
                            for study, target_size in target_sizes}
        for study, classifier in self.classifiers.items():
            self.add_module('classifier_%s' % study, classifier)
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.ndimension() == 2:
                nn.init.xavier_uniform(param)
            elif param.ndimension() == 1:
                param.data.fill_(0.)

    def forward(self, inputs):  # return_penalty=False):
        preds = {}
        for study, input in inputs.items():
            preds[study] = self.classifiers[study](self.embedder(input))
        return preds
        # if return_penalty:
        #     penalty = torch.sum(exp_pred[:, None, :] * (1 - exp_pred[:, None, :])
        #                         * latent[:, :, None] ** 2
        #                         * self.classifiers[study].weight.transpose(0, 1)[None, :, :] ** 2)
        #     penalty *= .5 * self.dropout.p / (1 - self.dropout.p) / latent.shape[0]
        #     penalties[study] = penalty
    #
    # def use_batch_norm(self, mode=True):
    #     # Foireux
    #     self._use_batch_norm = mode
    #     if not self._use_batch_norm:
    #         for batch_norm in self.batch_norms.values():
    #             batch_norm.eval()

    # def train(self, mode=True):
    #     super().train(mode)
    #     if not self._use_batch_norm:
    #         for batch_norm in self.batch_norms.values():
    #             batch_norm.eval()


class MultiStudyLoss(nn.Module):
    def __init__(self, study_weights: Dict[str, float], ) -> None:
        super().__init__()
        self.study_weights = study_weights

    def forward(self, preds: Dict[str, torch.FloatTensor],
                targets: Dict[str, torch.LongTensor]) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        loss = 0
        for study in preds:
            pred = preds[study]
            target = targets[study]
            loss += (F.nll_loss(pred, target, size_average=True)
                     * self.study_weights[study])
        return loss


class RandomChoiceIter:
    def __init__(self, choices, p, seed=None):
        self.random_state = check_random_state(seed)
        self.choices = choices
        self.p = p

    def __next__(self):
        return self.random_state.choice(self.choices, p=self.p)


class MultiStudyLoaderIter:
    def __init__(self, loader):
        studies = loader.studies
        loaders = {study: DataLoader(data,
                                     shuffle=True,
                                     batch_size=loader.batch_size,
                                     pin_memory=loader.cuda)
                   for study, data in studies.items()}
        self.loader_iters = {study: infinite_iter(loader)
                             for study, loader in loaders.items()}

        studies = list(studies.keys())
        self.sampling = loader.sampling
        if self.sampling == 'random':
            p = np.array([loader.study_weights[study] for study in studies])
            assert (np.all(p >= 0))
            p /= np.sum(p)
            self.study_iter = RandomChoiceIter(studies, p, loader.seed)
        elif self.sampling == 'cycle':
            self.study_iter = itertools.cycle(studies)
        elif self.sampling == 'all':
            self.studies = studies
        else:
            raise ValueError('Wrong value for `sampling`')

        self.cuda = loader.cuda
        self.device = loader.device

    def __next__(self):
        inputs, targets = {}, {}
        if self.sampling == 'all':
            for study in self.studies:
                input, target = next(self.loader_iters[study])
                if self.cuda:
                    input = input.cuda(device=self.device)
                    target = target.cuda(device=self.device)
                input, target = Variable(input), Variable(target)
                inputs[study], targets[study] = input, target
        else:
            study = next(self.study_iter)
            input, target = next(self.loader_iters[study])
            if self.cuda:
                input = input.cuda(device=self.device)
                target = target.cuda(device=self.device)
            input, target = Variable(input), Variable(target)
            inputs[study], targets[study] = input, target
        return inputs, targets


class MultiStudyLoader:
    def __init__(self, studies,
                 batch_size=128, sampling='cycle',
                 study_weights=None, seed=None,
                 cuda=False, device=-1):
        self.studies = studies
        self.batch_size = batch_size
        self.sampling = sampling
        self.study_weights = study_weights

        self.cuda = cuda
        self.device = device
        self.seed = seed

    def __iter__(self):
        return MultiStudyLoaderIter(self)


class MultiStudyClassifier(BaseEstimator):
    def __init__(self,
                 latent_size=30,
                 activation='linear',
                 batch_size=128, optimizer='sgd',
                 epoch_counting='all',
                 lr=0.001,
                 fine_tune=False,
                 dropout=0.5, input_dropout=0.25,
                 max_iter=10000, verbose=0,
                 device=-1,
                 sampling='cycle',
                 patience=100,
                 seed=None):

        self.latent_size = latent_size
        self.activation = activation
        self.input_dropout = input_dropout
        self.dropout = dropout

        self.sampling = sampling
        self.batch_size = batch_size

        self.optimizer = optimizer
        self.lr = lr

        self.fine_tune = fine_tune

        self.epoch_counting = epoch_counting
        self.patience = patience
        self.max_iter = max_iter

        self.verbose = verbose
        self.device = device
        self.seed = seed

    def fit(self, X, y, study_weights=None, callback=None):
        cuda, device = self._check_cuda()

        torch.manual_seed(self.seed)

        # Data
        X = {study: torch.from_numpy(this_X).float()
             for study, this_X in X.items()}
        y = {study: torch.from_numpy(this_y['contrast'].values).long()
             for study, this_y in y.items()}
        data = {study: TensorDataset(X[study], y[study]) for study in X}
        data_loader = MultiStudyLoader(data, sampling=self.sampling,
                                       batch_size=self.batch_size,
                                       seed=self.seed,
                                       study_weights=study_weights,
                                       cuda=cuda, device=device)
        # Model
        target_sizes = {study: int(this_y['contrast'].max()) + 1
                        for study, this_y in y.items()}
        in_features = next(iter(X.values())).shape[1]

        self.module_ = MultiStudyModule(
            in_features=in_features,
            activation=self.activation,
            latent_size=self.latent_size,
            input_dropout=self.input_dropout,
            dropout=self.dropout,
            target_sizes=target_sizes)
        self.module_.reset_parameters()

        # Loss function
        if study_weights is None or self.sampling == 'random':
            study_weights = {study: 1. for study in X}
        loss_function = MultiStudyLoss(study_weights)
        # Optimizers
        params = self.module_.parameters()
        if self.optimizer == 'adam':
            optimizer = Adam(params, lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = SGD(params, lr=self.lr, )
        else:
            raise ValueError

        # Verbosity + epoch counting
        if self.epoch_counting == 'all':
            n_samples = sum(len(this_X) for this_X in X.values())
        elif self.epoch_counting == 'target':
            if self.sampling == 'weighted_random':
                multiplier = (sum(study_weights.values()) /
                              next(iter(study_weights.values())))
            else:
                multiplier = len(X)
            n_samples = len(next(iter(X.values()))) * multiplier
        else:
            raise ValueError

        if self.verbose != 0:
            report_every = ceil(self.max_iter / self.verbose)
        else:
            report_every = None

        best_state = self.module_.state_dict()

        old_epoch = -1
        seen_samples = 0
        epoch_seen_samples = 0
        epoch_loss = 0
        best_loss = float('inf')
        no_improvement = 0
        for inputs, targets in data_loader:
            batch_size = sum(input.shape[0] for input in inputs.values())

            self.module_.train()
            optimizer.zero_grad()

            preds = self.module_(inputs)
            loss, study_losses = loss_function(preds, targets)
            loss.backward()
            optimizer.step()

            seen_samples += batch_size
            epoch_seen_samples += batch_size

            epoch_loss += this_loss.data[0] * batch_size

            epoch = floor(seen_samples / n_samples)
            if epoch > old_epoch:
                old_epoch = epoch
                epoch_loss /= epoch_seen_samples
                epoch_seen_samples = 0

                if (report_every is not None
                        and epoch % report_every == 0):
                    print('Epoch %.2f, train loss: %.4f'
                          % (epoch, epoch_loss))
                    if callback is not None:
                        callback(self, epoch)

                if epoch_loss > best_loss:
                    no_improvement += 1
                else:
                    no_improvement = 0
                    best_loss = epoch_loss
                    best_state = self.module_.state_dict()
                epoch_loss = 0

                if (no_improvement > self.patience
                        or epoch > self.max_iter):
                    print('Stopping at epoch %.2f, best train loss'
                          ' %.4f' % (epoch, best_loss))
                    self.module_.load_state_dict(best_state)
                    break
        X_red = {}
        self.module_.embedder.eval()
        for study, this_X in X.items():
            print('Fine tuning %s' % study)

            if cuda:
                this_X = this_X.cuda(device=device)
            this_X = Variable(this_X, volatile=True)
            X_red[study] = self.module_.embedder(this_X).data.cpu()
            data = TensorDataset(X_red[study], y[study])
            data_loader = DataLoader(data, shuffle=True,
                                     batch_size=self.batch_size,
                                     pin_memory=cuda)
            module = self.module_.classifiers[study]
            optimizer = Adam(module.parameters(), lr=self.lr)
            loss_function = F.nll_loss

            old_epoch = -1
            seen_samples = 0
            epoch_seen_samples = 0
            epoch_loss = 0
            best_loss = float('inf')
            no_improvement = 0
            for input, target in data_loader:
                batch_size = input.shape[0]
                if cuda:
                    input = input.cuda(device=device)
                    target = target.cuda(device=device)
                input = Variable(input)
                target = Variable(target)

                module.train()
                optimizer.zero_grad()
                pred = module(input)
                this_loss = loss_function(pred, target)
                this_loss.backward()
                optimizer.step()

                seen_samples += batch_size
                epoch_seen_samples += batch_size

                epoch_loss += this_loss.data[0] * batch_size

                epoch = floor(seen_samples / n_samples)

                if epoch > old_epoch:
                    old_epoch = epoch
                    epoch_loss /= epoch_seen_samples
                    epoch_seen_samples = 0

                    if (report_every is not None
                            and epoch % report_every == 0):
                        print('Epoch %.2f, train loss: %.4f'
                              % (epoch, epoch_loss))
                        if callback is not None:
                            callback(self, epoch)

                    if epoch_loss > best_loss:
                        no_improvement += 1
                    else:
                        no_improvement = 0
                        best_loss = epoch_loss
                        best_state = module.state_dict()
                    epoch_loss = 0

                    if (no_improvement > self.patience
                            or epoch > self.max_iter):
                        print('Stopping at epoch %.2f, best train loss'
                              ' %.4f' % (epoch, best_loss))
                        module.load_state_dict(best_state)
                        break



        # Fine tune with lbfgs
        # self.module_.eval()
        # loss_function = MultiStudyLoss(
        #     study_weights={study: 1. for study in X})
        # for study in X:
        #     this_X = torch.from_numpy(X[study]).float()
        #     this_target = torch.from_numpy(
        #         y[study]['contrast'].values).long()
        #     if cuda:
        #         this_X = this_X.cuda(device=device)
        #         this_target = this_target.cuda(device=device)
        #     this_X = {study: Variable(this_X)}
        #     this_target = {study: Variable(this_target)}
        #
        #     params = list(self.module_.classifiers[study].parameters())
        #
        #     optimizer = LBFGSScipy(params,
        #                            max_iter=100,
        #                            tolerance_grad=0,
        #                            tolerance_change=0,
        #                            report_every=2)
        #
        #     def closure():
        #         this_pred, this_penalty = self.module_(this_X,
        #                                                return_penalty=True)
        #         loss = loss_function(this_pred, this_target)
        #         loss += this_penalty[study]
        #         return loss
        #
        #     optimizer.step(closure)

        # Fine tune with scikit learn
        # for study in X:
        #     this_y = y[study]['contrast'].values
        #     this_X = self.predict_latent({study: X[study]})[study]
        #     scaler = StandardScaler()
        #     this_X = scaler.fit_transform(this_X)
        #     classifier = LogisticRegressionCV(solver='lbfgs',
        #                                       multi_class='multinomial',
        #                                       n_jobs=3,
        #                                       )
        #     classifier.fit(this_X, this_y)
        #
        #     # Assign to pytorch module
        #     self.module_.classifiers[study].weight.data \
        #         = torch.from_numpy(classifier.coef_).float()
        #     self.module_.classifiers[study].weight.bias \
        #         = torch.from_numpy(classifier.intercept_).float()
        #     self.module_.batch_norms[study] = Identity()
        #     self.module_.batch_norms[study].running_mean \
        #         = torch.from_numpy(scaler.mean_).float()
        #     self.module_.batch_norms[study].running_var \
        #         = torch.from_numpy(scaler.var_).float()
        callback(self, self.n_iter_)

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

    def predict_log_proba(self, X):
        cuda, device = self._check_cuda()
        self.module_.eval()
        X_ = {}
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            if cuda:
                this_X = this_X.cuda(device=device)
            X_[study] = Variable(this_X, volatile=True)
        preds = self.module_(X_)
        return {study: pred.data.cpu().numpy() for study, pred in
                preds.items()}

    def predict_latent(self, X):
        cuda, device = self._check_cuda()
        self.module_.eval()
        latents = {}
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            if cuda:
                this_X = this_X.cuda(device=device)
            this_X = Variable(this_X, volatile=True)
            latent = self.module_.embedder(this_X)
            latents[study] = latent.data.cpu().numpy()
        return latents

    def predict(self, X):
        preds = self.predict_log_proba(X)
        preds = {study: np.argmax(pred, axis=1)
                 for study, pred in preds.items()}
        dfs = {}
        for study in preds:
            pred = preds[study]
            dfs[study] = pd.DataFrame(
                dict(contrast=pred, study=0,
                     all_contrast=0,
                     subject=0))
        return dfs

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['module_']:
            if key in state:
                val = state.pop(key)
                with tempfile.SpooledTemporaryFile() as f:
                    torch.save(val, f)
                    f.seek(0)
                    state[key] = f.read()

        return state

    def __setstate__(self, state):
        disable_cuda = False
        for key in ['module_']:
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
