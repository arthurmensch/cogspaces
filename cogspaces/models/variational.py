import math
import tempfile
import warnings
from math import ceil, floor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset

from cogspaces.models.factored import Identity
from cogspaces.models.factored_fast import MultiStudyLoader, MultiStudyLoss, \
    MultiStudyModule

k1 = 0.63576
k2 = 1.87320
k3 = 1.48695


class VarLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 parametrization='logalpha'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # aka theta
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # aka log sigma2 / log alpha
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.parametrization = parametrization

        self.reset_parameters()

    @staticmethod
    def clip(input, to=8):

        input = input.masked_fill(input < -to, -to)
        input = input.masked_fill(input > to, to)
        return input

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.log_sigma2.data.fill_(0)
        self.log_sigma2.data += torch.log(self.weight.data ** 2)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.training:
            output = F.linear(input, self.weight, self.bias)
            std = torch.sqrt(F.linear(input ** 2,
                                      torch.exp(self.log_sigma2),
                                      None) + 1e-8)
            eps = Variable(torch.randn(*output.shape))
            return output + eps * std
        else:
            log_alpha = self.clip(self.log_sigma2 - torch.log(self.weight ** 2 + 1e-8))
            mask = log_alpha > 3
            print(mask.float().mean().data[0])
            output = F.linear(input, self.weight.masked_fill(mask, 0),
                              self.bias)
            return output

    def penalty(self):
        log_alpha = self.clip(self.log_sigma2 -
                              torch.log(self.weight ** 2 + 1e-8))
        return - (torch.sum(k1 * (F.sigmoid(k2 + k3 * log_alpha)
                                  - .5 * F.softplus(-log_alpha) - 1))
                  / self.in_features / self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'

    def reset_dropout(self, dropout):
        self.log_sigma2.data.fill_(0)
        self.log_sigma2.data += torch.log(self.weight.data ** 2)


class Embedder(nn.Module):
    def __init__(self, in_features, latent_size,
                 activation='linear'):
        super().__init__()

        self.linear = VarLinear(in_features, latent_size, bias=True, )

        if activation == 'linear':
            self.activation = Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, input):
        return self.activation(self.linear(input))

    def reset_parameters(self):
        self.linear.reset_parameters()

    def penalty(self):
        return self.linear.penalty()

    def reset_dropout(self, dropout):
        self.linear.reset_dropout(dropout)


class LatentClassifier(nn.Module):
    def __init__(self, latent_size, target_size):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(latent_size)
        self.linear = VarLinear(latent_size, target_size, bias=True, )

    def forward(self, input):
        input = self.batch_norm(input)
        return F.log_softmax(self.linear(input), dim=1)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.batch_norm.reset_parameters()

    def penalty(self):
        return self.linear.penalty()

    def reset_dropout(self, dropout):
        self.linear.reset_dropout(dropout)


class VarMultiStudyModule(nn.Module):
    def __init__(self, in_features,
                 latent_size,
                 target_sizes,
                 activation='linear'):
        super().__init__()

        self.embedder = Embedder(in_features, latent_size,
                                 activation=activation)

        self.classifiers = {study: LatentClassifier(latent_size, target_size,
                                                    )
                            for study, target_size in target_sizes.items()}
        for study, classifier in self.classifiers.items():
            self.add_module('classifier_%s' % study, classifier)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedder.reset_parameters()
        for classifier in self.classifiers.values():
            classifier.reset_parameters()

    def forward(self, inputs):
        preds = {}
        for study, input in inputs.items():
            preds[study] = self.classifiers[study](self.embedder(input))
        return preds

    def penalty(self, studies=None):
        penalty = self.embedder.penalty()
        if studies is not None:
            for study in studies:
                penalty += self.classifiers[study].penalty()
        return penalty

    def reset_dropout(self, input_dropout, dropout):
        self.embedder.reset_dropout(input_dropout)
        for classifier in self.classifiers.values():
            classifier.reset_dropout(dropout)


class VarMultiStudyClassifier(BaseEstimator):
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
                 variational=False,
                 sampling='cycle',
                 patience=50,
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

        self.variational = variational

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
        target_sizes = {study: int(this_y.max()) + 1
                        for study, this_y in y.items()}
        in_features = next(iter(X.values())).shape[1]

        phases = ['fixed_dropout', 'var_dropout']

        for phase in phases:
            if phase == 'fixed_dropout':
                self.module_ = MultiStudyModule(
                    in_features=in_features,
                    activation=self.activation,
                    latent_size=self.latent_size,
                    input_dropout=self.input_dropout,
                    dropout=self.dropout,
                    target_sizes=target_sizes)
                self.module_.reset_parameters()
                lr = self.lr
            else:
                old_module = self.module_
                self.module_ = VarMultiStudyModule(
                    in_features=in_features,
                    activation=self.activation,
                    latent_size=self.latent_size,
                    target_sizes=target_sizes)
                self.module_.load_state_dict(old_module.state_dict(),
                                             strict=False)
                lr = self.lr * .1

            if study_weights is None or self.sampling == 'random':
                study_weights = {study: 1. for study in X}
            loss_function = MultiStudyLoss(study_weights)
            # Optimizers
            if self.optimizer == 'adam':
                optimizer = Adam(self.module_.parameters(), lr=lr, )
            elif self.optimizer == 'sgd':
                optimizer = SGD(self.module_.parameters(), lr=lr, )
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
            epoch_loss = 0
            epoch_batch = 0
            best_loss = float('inf')
            no_improvement = 0
            for inputs, targets in data_loader:
                batch_size = sum(input.shape[0] for input in inputs.values())
                seen_samples += batch_size
                self.module_.train()
                optimizer.zero_grad()

                preds = self.module_(inputs)
                loss = loss_function(preds, targets)
                elbo = loss + self.module_.penalty()
                elbo.backward()
                optimizer.step()

                epoch_batch += 1
                epoch_loss *= (1 - 1 / epoch_batch)
                epoch_loss = loss.data[0] / epoch_batch

                epoch = floor(seen_samples / n_samples)
                if epoch > old_epoch:
                    old_epoch = epoch
                    epoch_batch = 0

                    if (report_every is not None
                            and epoch % report_every == 0):
                        penalty = self.module_.penalty()
                        print('Epoch %.2f, train loss: %.4f, penalty: %.4f'
                              % (epoch, epoch_loss, penalty))
                        weight = self.module_.embedder.linear.weight.data
                        density = (torch.sum(weight != 0)
                                   / weight.shape[0] / weight.shape[1])
                        print('Sparsity: %.4f' % (1 - density))
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
            # latent = self.module_.classifiers[study].batch_norm(latent)
            latents[study] = latent.data.cpu().numpy()
        return latents

    def predict_rec(self, X):
        cuda, device = self._check_cuda()
        self.module_.eval()
        recs = {}
        W = self.module_.embedder.linear.weight
        Ginv = torch.inverse(torch.matmul(W, W.transpose(0, 1)))
        back_proj = torch.matmul(Ginv, W)
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            if cuda:
                this_X = this_X.cuda(device=device)
            this_X = Variable(this_X, volatile=True)
            latent = self.module_.embedder(this_X)
            rec = torch.matmul(latent, back_proj)
            recs[study] = rec.data.cpu().numpy()
        return recs

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
