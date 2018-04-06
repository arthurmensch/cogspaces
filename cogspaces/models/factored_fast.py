import tempfile
import warnings
from math import ceil, floor
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

from cogspaces.models.factored import Identity


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
        return self.activation(self.linear(input))


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

        self.batch_norms = {study: nn.BatchNorm1d(latent_size)
                            for study in target_sizes}
        self.dropout = nn.Dropout(p=dropout)

        for study, size in target_sizes.items():
            self.classifiers[study] = nn.Linear(latent_size, size, bias=True)
            self.add_module('classifier_%s' % study, self.classifiers[study])
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.ndimension() == 2:
                nn.init.xavier_uniform(param)
            elif param.ndimension() == 1:
                param.data.fill_(0.)

    def forward(self, inputs):
        preds = {}
        for study, input in inputs:
            latent = self.embedder(input)
            latent = self.dropout(self.batch_norms[study](latent))
            preds[study] = F.log_softmax(self.classifiers[study](latent),
                                         dim=1)
        return preds


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
                   for study, data in studies.item()}
        self.loader_iters = infinite_iter(loaders)

        studies = list(studies.keys())
        self.sampling = loader.sampling
        if self.sampling == 'random':
            p = np.array([loader.study_weights[study] for study in studies])
            assert (np.all(p >= 0))
            p /= np.sum(p)
            self.study_iter = RandomChoiceIter(studies, p, loader.seed)
        elif self.sampling == 'cycle':
            self.study_iter = np.itertools.cycle(studies)
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
                input, target = next(self.loader_iters)
                if self.cuda:
                    input = input.cuda(device=self.device)
                    target = target.cuda(device=self.device)
                input, target = Variable(input), Variable(target)
                inputs[study], targets[study] = input, target
        else:
            study = next(self.study_iter)
            input, target = next(self.loader_iters)
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
        data = {study: TensorDataset(
            torch.from_numpy(X[study]),
            torch.from_numpy(y[study]['contrast'].values))
            for study in X}
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
        if study_weights is None:
            study_weights = {study: 1. for study in X}
        if self.sampling == 'weighted_random':
            loss_study_weights = {study: 1. for study in X}
        else:
            loss_study_weights = study_weights
        loss_function = MultiStudyLoss(study_weights=loss_study_weights)

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
        seen_samples = 0
        self.n_iter_ = 0
        old_epoch = -1

        if self.verbose != 0:
            report_every = ceil(self.max_iter / self.verbose)
        else:
            report_every = None

        phases = ['main', 'fine_tune']
        max_iter = {'main': self.max_iter, 'fine_tune': self.max_iter * 2}

        best_loss = float('inf')
        epoch_loss = 0
        epoch_seen_samples = 0
        no_improvement = 0

        best_state = {'module': self.module_.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      }

        for phase in phases:
            if phase == 'fine_tune':
                print('Fine tuning....')
                self.module_.embedder.input_dropout.p = 0.
                self.module_.embedder.linear.weight.requires_grad = False
                self.module_.embedder.linear.bias.requires_grad = False
            # Logging logic

            for inputs, targets in data_loader:
                self.module_.train()
                optimizer.zero_grad()

                batch_size = sum(input.shape[0] for input in inputs)
                preds = self.module_(inputs)
                this_loss = loss_function(preds, targets)
                this_loss.backward()
                optimizer.step()

                seen_samples += batch_size
                epoch_seen_samples += batch_size

                epoch_loss += this_loss.data[0] * batch_size

                self.n_iter_ = seen_samples / n_samples
                epoch = floor(self.n_iter_)

                if epoch > old_epoch:
                    old_epoch = epoch
                    epoch_loss /= epoch_seen_samples
                    epoch_seen_samples = 0

                    if (report_every is not None
                            and epoch % report_every == 0):
                        print('Epoch %.2f, train loss: %.4f'
                              % (epoch, epoch_loss))
                        if callback is not None:
                            callback(self, self.n_iter_)

                    if epoch_loss > best_loss:
                        no_improvement += 1
                    else:
                        no_improvement = 0
                        best_loss = epoch_loss
                        best_state = {'module': self.module_.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      }
                    epoch_loss = 0

                    if (no_improvement > self.patience
                            or epoch > max_iter[phase]):
                        print('Stopping at epoch %.2f, best train loss'
                              ' %.4f' % (epoch, best_loss))
                        self.module_.load_state_dict(best_state['module'])
                        optimizer.load_state_dict(best_state['optimizer'])
                        break
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
        X_ = {}
        for study, this_X in X.items():
            if cuda:
                this_X = this_X.cuda(device=device)
            X_[study] = Variable(this_X, volatile=True)
        self.module_.eval()
        preds = self.module_(X_)
        return {study: pred.cpu().numpy() for study, pred in preds.items()}

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
