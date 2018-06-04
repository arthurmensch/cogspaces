import math
from math import ceil, floor

import numpy as np
import pandas as pd
import tempfile
import torch
import torch.nn.functional as F
import warnings
from cogspaces.datasets.utils import get_output_dir
from cogspaces.models.factored import Identity
from cogspaces.models.factored_fast import MultiStudyLoader, MultiStudyLoss
from os.path import join, expanduser
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from torch import nn
from torch.nn import Parameter
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

k1 = 0.63576
k2 = 1.87320
k3 = 1.48695


class DropoutLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p=1e-8,
                 level='layer', var_penalty=0., adaptive=False,
                 init='normal',
                 sparsify=False):
        super().__init__(in_features, out_features, bias)

        assert p >= 1e-8
        self.p = p
        self.var_penalty = var_penalty

        if level == 'layer':
            self.log_alpha = Parameter(torch.Tensor(1, 1),
                                       requires_grad=adaptive)

        elif level == 'atom':
            self.log_alpha = Parameter(torch.Tensor(1, in_features),
                                       requires_grad=adaptive)
        elif level == 'coef':
            self.log_alpha = Parameter(torch.Tensor(out_features, in_features),
                                       requires_grad=adaptive)
        elif level == 'additive':
            assert adaptive
            self.log_sigma2 = Parameter(
                torch.Tensor(out_features, in_features),
                requires_grad=True)
        else:
            raise ValueError()

        self.sparsify = sparsify
        self.adaptive = adaptive
        self.level = level

        self.reset_dropout()

    def reset_parameters(self):
        if self.init == 'normal':
            super().reset_parameters()
        elif self.init == 'symmetric':
            super().reset_parameters()
            assign = np.load(expanduser(
                '~/work/repos/cogspaces/exps/assign_512.npy')).tolist()
            self.weight.data += self.weight.data[:, assign]
            self.weight.data /= 2
        elif self.init == 'orthogonal':
            nn.init.orthogonal_(self.weight.data,
                                gain=1 / math.sqrt(self.weight.shape[1]))
        elif self.init == 'loadings_128':
            self.weight.data = torch.from_numpy(
                np.load(expanduser('~/work/repos/cogspaces/exps/'
                                   'loadings_128.npy')))
        else:
            try:
                weight = np.load(self.init)
                self.weight.data = torch.from_numpy(weight)
            except:
                raise FileNotFoundError('Wrong init path %s' % self.init)
        if hasattr(self, 'level'):
            self.reset_dropout()

    def reset_dropout(self):
        log_alpha = math.log(self.p) - math.log(1 - self.p)

        if self.level != 'additive':
            self.log_alpha.data.fill_(log_alpha)
        else:
            self.log_sigma2.data = log_alpha + torch.log(
                self.weight.data ** 2 + 1e-8)

    def make_additive(self):
        assert self.level != 'additive'
        self.log_alpha.requires_grad = False
        self.level = 'additive'
        self.adaptive = True
        out_features, in_features = self.weight.shape
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features),
                                    requires_grad=True)
        self.log_sigma2.data = (self.log_alpha.expand(*self.weight.shape)
                                + torch.log(self.weight ** 2 + 1e-8)
                                ).detach()

        self.log_alpha.requires_grad = False
        print(np.array2string(self.get_p().detach().numpy(), precision=3))

    def make_non_adaptive(self):
        assert self.level != 'additive'
        self.adaptive = False
        self.log_alpha.requires_grad = False

    def make_adaptive(self):
        assert self.level != 'additive'
        self.adaptive = True
        self.log_alpha.requires_grad = True

    def get_var_weight(self):
        if self.level == 'additive':
            return torch.exp(self.log_sigma2)
            # return self.sigma ** 2
        else:
            return torch.exp(self.log_alpha) * self.weight ** 2

    def get_log_alpha(self):
        if self.level == 'additive':
            return torch.clamp(
                self.log_sigma2 - torch.log(self.weight ** 2 + 1e-8), -8, 8)
        else:
            return torch.clamp(self.log_alpha, -8, 8)

    def get_p(self):
        return 1 / (1 + torch.exp(-self.get_log_alpha()))

    def forward(self, input):
        if self.training:
            if self.adaptive:
                output = F.linear(input, self.weight, self.bias)
                # Local reparemtrization trick: gaussian dropout noise on input
                # <-> gaussian noise on output
                std = torch.sqrt(
                    F.linear(input ** 2, self.get_var_weight(), None) + 1e-8)
                eps = torch.randn_like(output, requires_grad=False)
                return output + std * eps
            else:
                eps = torch.randn_like(input, requires_grad=False)
                input = input * (
                        1 + torch.exp(.5 * self.get_log_alpha()) * eps)
                return F.linear(input, self.weight, self.bias)
        else:
            if self.sparsify:
                weight = self.sparse_weight
            else:
                weight = self.weight
            return F.linear(input, weight, self.bias)

    def penalty(self):
        if not self.adaptive or self.var_penalty == 0:
            return torch.tensor(0., device=self.weight.device,
                                dtype=torch.float)
        else:
            log_alpha = self.get_log_alpha()
            var_penalty = - k1 * (F.sigmoid(k2 + k3 * log_alpha)
                                  - .5 * F.softplus(-log_alpha)
                                  - 1).expand(*self.weight.shape).sum()
            return var_penalty * self.var_penalty

    @property
    def density(self):
        return (self.sparse_weight != 0).float().mean().item()

    @property
    def sparse_weight(self):
        mask = self.get_log_alpha() > 1
        return self.weight.masked_fill(mask, 0)


class Embedder(nn.Module):
    def __init__(self, in_features, latent_size, var_penalty,
                 activation='linear', dropout=0., adaptive=False,
                 init='gamble_rest'):
        super().__init__()

        self.init = init

        self.linear = DropoutLinear(in_features, latent_size,
                                    adaptive=adaptive,
                                    var_penalty=var_penalty,
                                    sparsify=False,
                                    level='additive' if adaptive else 'layer',
                                    p=dropout, bias=True)
        if activation == 'linear':
            self.activation = Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        self.reset_parameters()

    def forward(self, input):
        return self.activation(self.linear(input))

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.linear.reset_dropout()

    def penalty(self):
        return self.linear.penalty()


class LatentClassifier(nn.Module):
    def __init__(self, latent_size, target_size, var_penalty,
                 dropout=0., adaptive=False):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(latent_size)
        self.linear = DropoutLinear(latent_size,
                                    target_size, bias=True, p=dropout,
                                    var_penalty=var_penalty,
                                    adaptive=adaptive,
                                    level='layer')

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
                 lengths,
                 activation='linear',
                 input_dropout=0.,
                 regularization=1.,
                 latent_dropout=0.,
                 adaptive='embedding+classifier',
                 ):
        super().__init__()

        embedder_adaptive = 'embedding' in adaptive

        total_length = sum(list(lengths.values()))
        self.embedder = Embedder(in_features, latent_size,
                                 dropout=input_dropout,
                                 adaptive=embedder_adaptive,
                                 var_penalty=regularization / total_length,
                                 activation=activation)
        classifier_adaptive = 'classifier' in adaptive
        self.classifiers = {study: LatentClassifier(
            latent_size, target_size, dropout=latent_dropout,
            var_penalty=regularization / lengths[study],
            adaptive=classifier_adaptive, )
            for study, target_size in target_sizes.items()}
        for study, classifier in self.classifiers.items():
            self.add_module('classifier_%s' % study, classifier)

    def reset_parameters(self):
        self.embedder.reset_parameters()
        for classifier in self.classifiers.values():
            classifier.reset_parameters()

    def forward(self, inputs):
        preds = {}
        for study, input in inputs.items():
            preds[study] = self.classifiers[study](self.embedder(input))
        return preds

    def penalty(self, inputs):
        return (self.embedder.penalty()
                + sum(self.classifiers[study].penalty()
                      for study in inputs))


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


class DeviceWarning(object):
    pass


class VarMultiStudyClassifier(BaseEstimator):
    def __init__(self, latent_size=30,
                 activation='linear',
                 batch_size=128, optimizer='sgd',
                 epoch_counting='all',
                 lr=0.001,
                 dropout=0.5, input_dropout=0.25,
                 max_iter=10000, verbose=0,
                 device=-1,
                 regularization=1.,
                 weight_power=0.5,
                 variational=False,
                 sampling='cycle',
                 rotation=False,
                 n_jobs=1,
                 patience=200,
                 seed=None):

        self.latent_size = latent_size
        self.activation = activation
        self.input_dropout = input_dropout
        self.dropout = dropout

        self.regularization = regularization

        self.rotation = rotation

        self.sampling = sampling
        self.batch_size = batch_size

        self.optimizer = optimizer
        self.lr = lr

        self.weight_power = weight_power

        self.variational = variational

        self.epoch_counting = epoch_counting
        self.patience = patience
        self.max_iter = max_iter

        self.verbose = verbose
        self.device = device
        self.seed = seed

        self.n_jobs = n_jobs

    def fit(self, X, y, callback=None):
        torch.set_num_threads(self.n_jobs)
        device = self._check_device()

        torch.manual_seed(self.seed)
        # Data
        X = {study: torch.from_numpy(this_X).float()
             for study, this_X in X.items()}
        y = {study: torch.from_numpy(this_y['contrast'].values).long()
             for study, this_y in y.items()}
        data = {study: TensorDataset(X[study], y[study]) for study in X}
        lengths = {study: len(this_data)
                   for study, this_data in data.items()}
        lengths_arr = np.array(list(lengths.values()))
        total_length = np.sum(lengths_arr)
        study_weights = np.float_power(lengths_arr, self.weight_power)
        study_weights /= np.sum(study_weights)
        study_weights = {study: study_weight for study, study_weight
                         in zip(lengths, study_weights)}
        if self.sampling == 'random':
            eff_lengths = {study: total_length * study_weight for
                           study, study_weight
                           in study_weights.items()}
        else:
            eff_lengths = lengths

        data_loader = MultiStudyLoader(data, sampling=self.sampling,
                                       batch_size=self.batch_size,
                                       seed=self.seed,
                                       study_weights=study_weights,
                                       device=device)
        # Model
        target_sizes = {study: int(this_y.max()) + 1
                        for study, this_y in y.items()}
        in_features = next(iter(X.values())).shape[1]

        # Loss
        if self.sampling == 'random':
            loss_study_weights = {study: 1. for study in X}
        else:
            loss_study_weights = study_weights
        loss_function = MultiStudyLoss(loss_study_weights, )

        module = self.module_ = VarMultiStudyModule(
            in_features=in_features,
            input_dropout=self.input_dropout,
            latent_dropout=self.dropout,
            adaptive='classifier',
            regularization=self.regularization,
            activation=self.activation,
            lengths=eff_lengths,
            latent_size=self.latent_size,
            target_sizes=target_sizes)

        # Optimizers
        # optimizer = Adam(filter(lambda p: p.requires_grad,
        #                         module.parameters()),
        #                  lr=self.lr, amsgrad=True)

        # Verbosity + epoch counting
        if self.epoch_counting == 'all':
            n_samples = sum(len(this_X) for this_X in X.values())
        elif self.epoch_counting == 'target':
            if self.sampling == 'random':
                multiplier = (sum(study_weights.values()) /
                              next(iter(study_weights.values())))
            else:
                multiplier = len(X)
            n_samples = len(next(iter(X.values()))) * multiplier
        else:
            raise ValueError

        random_state = check_random_state(0)
        size = in_features * self.latent_size
        indices = random_state.permutation(size)[:100]
        self.recorded_ = []

        for phase in ['pretrain', 'sparsify']:
            if self.max_iter[phase] == 0:
                continue
            if self.verbose != 0:
                report_every = ceil(self.max_iter[phase] / self.verbose)
            else:
                report_every = None

            print('Phase :', phase)
            print('------------------------------')
            if phase == 'pretrain':
                optimizer = Adam(filter(lambda p: p.requires_grad,
                                        module.parameters()),
                                 lr=self.lr, amsgrad=True)
            else:  # phase == 'sparsify':
                module.embedder.linear.make_additive()
                optimizer = Adam(filter(lambda p: p.requires_grad,
                                        module.parameters()),
                                 lr=self.lr, amsgrad=True)

            best_state = module.state_dict()

            old_epoch = -1
            epoch = 0
            seen_samples = 0
            epoch_loss = float('inf')
            epoch_batch = 0
            best_loss = float('inf')
            no_improvement = 0
            epoch_penalty = 0
            for inputs, targets in data_loader:
                if epoch > old_epoch:
                    old_epoch = epoch
                    epoch_batch = 0
                    self.recorded_.append(module.embedder.linear.weight
                                          .detach().view((-1))[
                                              indices].numpy())
                    if (report_every is not None
                            and epoch % report_every == 0):
                        density = module.embedder.linear.density
                        print('Epoch %.2f, train loss: %.4f, penalty: %.4f,'
                              ' density: %.4f'
                              % (epoch, epoch_loss, epoch_penalty, density))
                        p = {}
                        for study, classifier in self.module_.classifiers.items():
                            p[
                                study] = classifier.linear.get_p().detach().numpy()
                        print('Dropout', ' '.join('%s: %.2f'
                                                  % (study, p[study]) for
                                                  study in p))
                        p = module.embedder.linear.get_p().detach().numpy()
                        print('Input dropout', np.array2string(p, precision=3,
                                                               suppress_small=True,
                                                               edgeitems=5))
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
                            or epoch >= self.max_iter[phase]):
                        print('Stopping at epoch %.2f, train loss'
                              ' %.4f' % (epoch, epoch_loss))
                        module.load_state_dict(best_state)
                        torch.save(module, join(get_output_dir(),
                                                'model_%s.pkl' % phase))
                        print('-----------------------------------')
                        break

                batch_size = sum(input.shape[0] for input in inputs.values())
                seen_samples += batch_size
                optimizer.zero_grad()
                module.train()
                preds = module(inputs)
                loss = loss_function(preds, targets)
                penalty = module.penalty(inputs)
                # penalty = module.penalty(inputs)
                loss += penalty
                loss.backward()
                optimizer.step()

                epoch_batch += 1
                epoch_loss *= (1 - 1 / epoch_batch)
                epoch_loss += loss.item() / epoch_batch

                epoch_penalty *= (1 - 1 / epoch_batch)
                epoch_penalty += penalty.item() / epoch_batch

                epoch = floor(seen_samples / n_samples)
        # self.recorded_ = np.concatenate([record[None, :] for
        #                                  record in self.recorded_], axis=0)
        print('Final density %s' % module.embedder.linear.density)

        if self.max_iter['finetune'] > 0:
            if self.verbose != 0:
                report_every = ceil(self.max_iter['finetune'] / self.verbose)
            else:
                report_every = None
            X_red = {}
            for study, this_X in X.items():
                print('Fine tuning %s' % study)
                this_X = this_X.to(device=device)
                with torch.no_grad():
                    self.module_.embedder.linear.sparsify = True
                    self.module_.embedder.eval()
                    X_red[study] = self.module_.embedder(this_X).to(
                        device=device)
                data = TensorDataset(X_red[study], y[study])
                data_loader = DataLoader(data, shuffle=True,
                                         batch_size=self.batch_size,
                                         drop_last=False,
                                         pin_memory=device.type == 'cuda')
                this_module = module.classifiers[study]
                if this_module.linear.adaptive:
                    this_module.linear.make_non_adaptive()
                    # this_module.linear.log_alpha.fill_(np.log(self.dropout /
                    #                                           (1 - self.dropout)
                    #                                           ))
                optimizer = Adam(filter(lambda p: p.requires_grad,
                                        this_module.parameters()),
                                 lr=self.lr, amsgrad=True)
                loss_function = F.nll_loss

                seen_samples = 0
                best_loss = float('inf')
                no_improvement = 0
                epoch = 0
                best_state = module.state_dict()
                for epoch in range(self.max_iter['finetune']):
                    epoch_batch = 0
                    epoch_penalty = 0
                    epoch_loss = 0
                    for input, target in data_loader:
                        batch_size = input.shape[0]
                        input = input.to(device=device)
                        target = target.to(device=device)

                        module.train()
                        optimizer.zero_grad()
                        this_module.train()
                        pred = this_module(input)
                        loss = loss_function(pred, target)
                        penalty = this_module.penalty()
                        loss += penalty
                        loss.backward()
                        optimizer.step()

                        seen_samples += batch_size
                        epoch_batch += 1
                        epoch_loss *= (1 - epoch_batch)
                        epoch_loss = loss.item() / epoch_batch
                        epoch_penalty *= (1 - 1 / epoch_batch)
                        epoch_penalty += penalty.item() / epoch_batch

                    if (report_every is not None
                            and epoch % report_every == 0):
                        print(
                            'Epoch %.2f, train loss: %.4f, penalty: %.4f, p: %.2f'
                            % (epoch, epoch_loss, epoch_penalty,
                               this_module.linear.get_p().item()))

                    if epoch_loss > best_loss:
                        no_improvement += 1
                    else:
                        no_improvement = 0
                        best_loss = epoch_loss
                        best_state = module.state_dict()
                    if no_improvement > self.patience:
                        break
                module.load_state_dict(best_state)
                print('Stopping at epoch %.2f, train loss'
                      ' %.4f' % (epoch, epoch_loss))
                print('-----------------------------------')
        if callback is not None:
            callback(self, epoch)
        return self

    def _check_device(self):
        if self.device == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:%i' % self.device)
        return device

    def predict_log_proba(self, X):
        device = self._check_device()
        self.module_.eval()
        X_ = {}
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            this_X = this_X.to(device=device)
            X_[study] = this_X
        with torch.no_grad():
            preds = self.module_(X_)
        return {study: pred.data.cpu().numpy() for study, pred in
                preds.items()}

    def predict_latent(self, X):
        device = self._check_device()
        self.module_.eval()
        latents = {}
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            this_X = this_X.to(device=device)
            with torch.no_grad():
                latent = self.module_.embedder(this_X)
            # latent = self.module_.classifiers[study].batch_norm(latent)
            latents[study] = latent.data.cpu().numpy()
        return latents

    def predict_rec(self, X):
        device = self._check_device()
        self.module_.eval()
        recs = {}
        W = self.module_.embedder.linear.weight
        Ginv = torch.inverse(torch.matmul(W, W.transpose(0, 1)))
        back_proj = torch.matmul(Ginv, W)
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            this_X = this_X.to(device=device)
            with torch.no_grad():
                latent = self.module_.embedder(this_X)
                rec = torch.matmul(latent, back_proj)
            recs[study] = rec.cpu().numpy()
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
        def uses_cuda(device):
            return False
            if isinstance(device, torch.device):
                device = device.type
            return device.startswith('cuda')

        disable_cuda = False
        for key in ['module_']:
            if key not in state:
                continue
            dump = state.pop(key)
            with tempfile.SpooledTemporaryFile() as f:
                f.write(dump)
                f.seek(0)
                if (
                        uses_cuda(state['device']) and
                        not torch.cuda.is_available()
                ):
                    disable_cuda = True
                    val = torch.load(
                        f, map_location=lambda storage, loc: storage)
                else:
                    val = torch.load(f)
            state[key] = val
        if disable_cuda:
            warnings.warn(
                "Model configured to use CUDA but no CUDA devices "
                "available. Loading on CPU instead.",
                DeviceWarning)
            state['device'] = 'cpu'

        self.__dict__.update(state)
