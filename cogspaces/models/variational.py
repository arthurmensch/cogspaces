import math
import tempfile
import warnings
from math import ceil, floor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from os.path import join
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

from cogspaces.datasets.utils import get_output_dir
from cogspaces.models.factored import Identity
from cogspaces.models.factored_fast import MultiStudyLoader, MultiStudyLoss
from cogspaces.utils.dict_learning import dict_learning

k1 = 0.63576
k2 = 1.87320
k3 = 1.48695


class DropoutLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p=0.):
        super().__init__(in_features, out_features, bias)
        self.p = p

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'log_alpha'):
            self.reset_dropout()

    def reset_dropout(self):
        pass

    def forward(self, input):
        if self.training:
            # Local reparemtrization trick: gaussian dropout noise on input
            # <-> gaussian noise on output
            p = max(self.p, 1e-8)
            std = math.sqrt(p / (1 - p))
            eps = Variable(torch.randn(*input.shape))
            input = input * (1 + std * eps)
            return F.linear(input, self.weight, self.bias)
        else:
            return super().forward(input)

    def penalty(self):
        return Variable(torch.FloatTensor([0]))

    @property
    def density(self):
        return 1

    @property
    def sparse_weight(self):
        return self.weight


class AdaDropoutLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p=0.,
                 level='layer'):
        super().__init__(in_features, out_features, bias)
        self.p = p

        if level == 'layer':
            self.log_alpha = Parameter(torch.Tensor(1, 1))
        elif level == 'atom':
            self.log_alpha = Parameter(torch.Tensor(out_features, 1))
        elif level == 'coef':
            self.log_alpha = Parameter(torch.Tensor(out_features, in_features))
        else:
            raise ValueError()
        self.reset_dropout()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'log_alpha'):
            self.reset_dropout()

    def reset_dropout(self):
        p = max(self.p, 1e-8)
        log_alpha = math.log(p) - math.log(1 - p)
        self.log_alpha.data.fill_(log_alpha)

    def forward(self, input):
        if self.training:
            output = super().forward(input)
            # Local reparemtrization trick: gaussian dropout noise on input
            # <-> gaussian noise on output
            std = torch.sqrt(F.linear(input ** 2,
                                      torch.exp(self.log_alpha)
                                      * self.weight ** 2,
                                      None) + 1e-8)
            eps = Variable(torch.randn(*output.shape))
            return output + std * eps
        else:
            mask = self.log_alpha > 3
            output = F.linear(input, self.weight.masked_fill(mask, 0),
                              self.bias)
            return output

    def penalty(self):
        log_alpha = self.log_alpha
        return - k1 * (F.sigmoid(k2 + k3 * log_alpha)
                       - .5 * F.softplus(-log_alpha)
                       - 1).expand(*self.weight.shape).sum()

    @property
    def density(self):
        return (self.sparse_weight != 0).float().mean().data[0]

    @property
    def sparse_weight(self):
        mask = self.log_alpha.expand(*self.weight.shape) > 3
        return self.weight.masked_fill(mask, 0)


class AdditiveAdaDropoutLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p=0.):
        super().__init__(in_features, out_features, bias)
        self.p = p
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features))
        self.reset_dropout()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'log_sigma2'):
            self.reset_dropout()

    @property
    def log_alpha(self):
        return self.log_sigma2 - torch.log(self.weight ** 2 + 1e-8)

    def reset_dropout(self):
        p = max(self.p, 1e-8)
        log_alpha = math.log(p) - math.log(1 - p)
        self.log_sigma2.data = log_alpha + torch.log(self.weight.data ** 2
                                                     + 1e-8)

    def forward(self, input):
        if self.training:
            weight = self.weight
            var_weight = torch.exp(self.log_sigma2)
            # Local reparametrization trick: gaussian dropout noise on input
            # <-> gaussian noise on output
            output = F.linear(input, weight, self.bias)
            std = torch.sqrt(F.linear(input ** 2, var_weight, None) + 1e-8)
            eps = Variable(torch.randn(*output.shape))
            return output + eps * std
        else:
            mask = self.log_alpha > 3
            output = F.linear(input, self.weight.masked_fill(mask, 0),
                              self.bias)
            return output

    def penalty(self):
        log_alpha = self.log_alpha
        return - torch.sum(k1 * (F.sigmoid(k2 + k3 * log_alpha)
                                 - .5 * F.softplus(-log_alpha) - 1))

    @property
    def density(self):
        return (self.sparse_weight != 0).float().mean().data[0]

    @property
    def sparse_weight(self):
        mask = self.log_alpha.expand(*self.weight.shape) > 3
        return self.weight.masked_fill(mask, 0)


class Embedder(nn.Module):
    def __init__(self, in_features, latent_size,
                 activation='linear', dropout=0., adaptive=False):
        super().__init__()

        if adaptive:
            self.linear = AdditiveAdaDropoutLinear(in_features,
                                                   latent_size, bias=True,
                                                   p=dropout)
        else:
            self.linear = DropoutLinear(in_features, latent_size, bias=True)
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


class LatentClassifier(nn.Module):
    def __init__(self, latent_size, target_size, dropout=0.,
                 adaptive=False):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(latent_size)
        if adaptive:
            self.linear = AdaDropoutLinear(latent_size,
                                           target_size, bias=True, p=dropout,
                                           level='layer')
        else:
            self.linear = DropoutLinear(latent_size, target_size, bias=True,
                                        p=dropout)

    def forward(self, input):
        input = self.batch_norm(input)
        return F.log_softmax(self.linear(input), dim=1)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.batch_norm.reset_parameters()

    def penalty(self):
        return self.linear.penalty()


class VarMultiStudyModule(nn.Module):
    def __init__(self, in_features,
                 latent_size,
                 target_sizes,
                 activation='linear',
                 input_dropout=0.,
                 latent_dropout=0.,
                 adaptivity='embedding+classifier',
                 ):
        super().__init__()

        embedder_adaptive = 'embedding' in adaptivity
        self.embedder = Embedder(in_features, latent_size,
                                 dropout=input_dropout,
                                 adaptive=embedder_adaptive,
                                 activation=activation)
        classifier_adaptive = 'classifier' in adaptivity
        self.classifiers = {study: LatentClassifier(
            latent_size, target_size, dropout=latent_dropout,
            adaptive=classifier_adaptive, )
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
        classifier_penalties = {study: self.classifiers[study].penalty()
                                for study in studies}
        return self.embedder.penalty(), classifier_penalties


def regularization_schedule(start_value, stop_value, warmup, cooldown,
                            max_iter, epoch):
    warmup_epoch = floor(warmup * max_iter)
    cooldown_epoch = floor(cooldown * max_iter)
    if epoch <= warmup_epoch:
        return start_value
    elif epoch >= cooldown_epoch:
        return stop_value
    else:
        alpha = (epoch - warmup_epoch) / (cooldown_epoch - warmup_epoch)
        return start_value * (1 - alpha) + stop_value * alpha


class VarMultiStudyClassifier(BaseEstimator):
    def __init__(self, latent_size=30,
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
                 rotation=False,
                 patience=200,
                 regularization=0.,
                 l1_penalty=1e-2,
                 seed=None):

        self.latent_size = latent_size
        self.activation = activation
        self.input_dropout = input_dropout
        self.dropout = dropout

        self.rotation = rotation

        self.l1_penalty = l1_penalty

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

        self.regularization = regularization

    def fit(self, X, y, study_weights=None, callback=None):
        cuda, device = self._check_cuda()

        torch.manual_seed(self.seed)

        # Data
        X = {study: torch.from_numpy(this_X).float()
             for study, this_X in X.items()}
        y = {study: torch.from_numpy(this_y['contrast'].values).long()
             for study, this_y in y.items()}
        data = {study: TensorDataset(X[study], y[study]) for study in X}
        lengths = {study: len(this_data)
                   for study, this_data in data.items()}
        total_length = sum(iter(lengths.values()))
        all_studies = list(data.keys())
        data_loader = MultiStudyLoader(data, sampling=self.sampling,
                                       batch_size=self.batch_size,
                                       seed=self.seed,
                                       study_weights=study_weights,
                                       cuda=cuda, device=device)
        # Model
        target_sizes = {study: int(this_y.max()) + 1
                        for study, this_y in y.items()}
        in_features = next(iter(X.values())).shape[1]

        # Loss
        if study_weights is None or self.sampling == 'random':
            loss_study_weights = {study: 1. for study in X}
        else:
            loss_study_weights = study_weights
        loss_function = MultiStudyLoss(loss_study_weights, )

        modules = {
            'pretrain': VarMultiStudyModule(
                in_features=in_features,
                input_dropout=self.input_dropout,
                latent_dropout=self.dropout,
                adaptivity='classifier',
                activation=self.activation,
                latent_size=self.latent_size,
                target_sizes=target_sizes),
            'sparsify': VarMultiStudyModule(
                in_features=in_features,
                input_dropout=self.input_dropout,
                latent_dropout=self.dropout,
                adaptivity='embedding+classifier',
                activation=self.activation,
                latent_size=self.latent_size,
                target_sizes=target_sizes),
            'finetune': VarMultiStudyModule(
                in_features=in_features,
                activation=self.activation,
                latent_size=self.latent_size,
                input_dropout=0,
                adaptivity='embedding',
                latent_dropout=self.dropout,
                target_sizes=target_sizes, )}

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

        for phase in ['pretrain', 'sparsify']:
            self.module_ = module = modules[phase]
            if phase == 'pretrain':
                lr = self.lr
                # modules[phase] = torch.load(join(get_output_dir(),
                #                                 'model_%s.pkl' % phase))
                # continue
            else:
                # modules[phase] = torch.load(join(get_output_dir(),
                #                                 'model_%s.pkl' % phase))
                # continue
                module.load_state_dict(modules['pretrain'].state_dict(),
                                       strict=False)
                module.embedder.linear.reset_dropout()
                lr = self.lr * .1
            # Optimizers
            if self.optimizer == 'adam':
                optimizer = Adam(module.parameters(), lr=lr, )
            elif self.optimizer == 'sgd':
                optimizer = SGD(module.parameters(), lr=lr, )
            else:
                raise ValueError

            best_state = module.state_dict()

            old_epoch = -1
            epoch = 0
            seen_samples = 0
            epoch_loss = 0
            epoch_batch = 0
            best_loss = float('inf')
            no_improvement = 0
            epoch_penalty = 0
            for inputs, targets in data_loader:
                batch_size = sum(input.shape[0] for input in inputs.values())
                seen_samples += batch_size
                module.train()
                optimizer.zero_grad()

                preds = module(inputs)
                loss = loss_function(preds, targets)
                embedder_penalty, penalties = module.penalty(inputs)

                penalty = embedder_penalty / total_length * self.regularization
                penalty += sum(this_penalty / lengths[study]
                               for study, this_penalty
                               in penalties.items())
                penalty += self.l1_penalty * sum(torch.sum(
                    torch.abs(module.classifiers[study].linear.weight)) for
                                      study in inputs)
                loss += penalty
                loss.backward()
                optimizer.step()
                if hasattr(module.embedder.linear, 'log_sigma2'):
                    module.embedder.linear.log_sigma2.data = (
                            torch.clamp(module.embedder.linear.log_alpha.data,
                                        min=-8, max=8) +
                            torch.log(module.embedder.linear.weight.data ** 2
                                      + 1e-8))

                epoch_batch += 1
                epoch_loss *= (1 - 1 / epoch_batch)
                epoch_loss += loss.data[0] / epoch_batch

                epoch_penalty *= (1 - 1 / epoch_batch)
                epoch_penalty += penalty.data[0] / epoch_batch

                epoch = floor(seen_samples / n_samples)
                if epoch > old_epoch:
                    old_epoch = epoch
                    epoch_batch = 0
                    if (report_every is not None
                            and epoch % report_every == 0):
                        penalty *= self.regularization
                        density = module.embedder.linear.density
                        print('Epoch %.2f, train loss: %.4f, penalty: %.4f,'
                              ' density: %.4f'
                              % (epoch, epoch_loss, epoch_penalty, density))
                        p = {}
                        for study, classifier in self.module_.classifiers.items():
                            log_alpha = classifier.linear.log_alpha.data[0]
                            p[study] = 1 / (1 + math.exp(- log_alpha))
                        print('Dropout', ' '.join('%s: %.2f'
                                                  % (study, this_p) for
                                                  study, this_p in
                                                  p.items()))
                        if phase in 'sparsify':
                            snr = torch.exp(
                                -.5 * module.embedder.linear.log_alpha)
                            print(snr.data.numpy())
                        if callback is not None:
                            callback(self, epoch)

                    if epoch_loss > best_loss:
                        no_improvement += 1
                    else:
                        no_improvement = 0
                        best_loss = epoch_loss
                        best_state = module.state_dict()
                    epoch_loss = 0
                    epoch_penalty = 0

                    if (no_improvement > self.patience
                            or epoch > self.max_iter):
                        print('Stopping at epoch %.2f, best train loss'
                              ' %.4f' % (epoch, best_loss))
                        callback(self, epoch)
                        module.load_state_dict(best_state)
                        torch.save(module, join(get_output_dir(),
                                                'model_%s.pkl' % phase))
                        print('-----------------------------------')
                        break

        callback(self, self.max_iter)

        self.module_sparsify_ = modules['sparsify']
        self.module_ = modules['finetune']
        self.module_.load_state_dict(modules['sparsify'].state_dict(),
                                     strict=False)
        # self.module_.embedder.linear.weight.data = \
        #     modules['sparsify'].embedder.linear.sparse_weight.data

        if self.rotation:
            classif = []
            classif_weights = []
            for study, classifier in self.module_.classifiers.items():
                these_classif = classifier.linear.weight.data.numpy()
                these_classif_weights = np.ones(these_classif.shape[0])
                these_classif_weights /= these_classif.shape[0]
                these_classif_weights *= math.sqrt(lengths[study])
                classif.append(classifier.linear.weight.data.numpy())
                classif_weights.append(these_classif_weights)
            classif = np.concatenate(classif, axis=0)
            classif_weights = np.concatenate(classif_weights, axis=0)
            classif_weights /= np.sum(classif_weights) / len(classif_weights)
            classif = StandardScaler().fit_transform(classif)
            code, dictionary, errors = dict_learning(classif,
                                                     method='lars',
                                                     alpha=.01, max_iter=10000,
                                                     sample_weights=classif_weights,
                                                     n_components=128,
                                                     verbose=True)
            self.module_.embedder.linear.weight.data = \
                torch.FloatTensor(dictionary) @ self.module_.embedder.linear.weight.data
            print(errors[-1], (code == 0).astype('float').mean())

        for study in self.module_.classifiers:
            classifier = self.module_.classifiers[study]
            log_alpha = \
                modules['sparsify'].classifiers[study].linear.log_alpha.data[0]
            p = 1 / (1 + math.exp(-log_alpha))
            p = min(p, 0.75)
            classifier.linear.p = 0
        nnz = self.module_.embedder.linear.weight != 0
        density = nnz.float().mean().data[0]
        print('Final density %s' % density)
        lr = self.lr
        X_red = {}
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
            optimizer = Adam(module.parameters(), lr=lr)
            loss_function = F.nll_loss

            seen_samples = 0
            best_loss = float('inf')
            no_improvement = 0
            epoch = 0
            for epoch in range(self.max_iter):
                epoch_batch = 0
                epoch_loss = 0
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
                    loss = loss_function(pred, target)
                    penalty = 0
                    elbo = loss + penalty
                    elbo.backward()
                    optimizer.step()

                    seen_samples += batch_size
                    epoch_batch += 1
                    epoch_loss *= (1 - 1 / epoch_batch)
                    epoch_loss = loss.data[0] / epoch_batch

                if (report_every is not None
                        and epoch % report_every == 0):
                    print('Epoch %.2f, train loss: %.4f, penalty: %.4f'
                          % (epoch, epoch_loss, 0))

                if epoch_loss > best_loss:
                    no_improvement += 1
                else:
                    no_improvement = 0
                    best_loss = epoch_loss
                    best_state = module.state_dict()

                if no_improvement > self.patience:
                    break
            # module.load_state_dict(best_state)
            print('Stopping at epoch %.2f, best train loss'
                  ' %.4f' % (epoch, best_loss))
            print('-----------------------------------')
        callback(self, epoch)
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
        for key in ['module_', 'module_sparsify_']:
            if key in state:
                val = state.pop(key)
                with tempfile.SpooledTemporaryFile() as f:
                    torch.save(val, f)
                    f.seek(0)
                    state[key] = f.read()

        return state

    def __setstate__(self, state):
        disable_cuda = False
        for key in ['module_', 'module_sparsify_']:
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
