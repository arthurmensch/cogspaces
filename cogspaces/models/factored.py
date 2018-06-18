import math
from math import ceil, floor

import numpy as np
import pandas as pd
import tempfile
import torch
import torch.nn.functional as F
import warnings
from joblib import load
from sklearn.base import BaseEstimator
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from cogspaces.data import MultiStudyLoader
from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.modules.linear import DropoutLinear
from cogspaces.modules.loss import MultiStudyLoss
from cogspaces.modules.utils import Identity


class Embedder(nn.Module):
    def __init__(self, in_features, latent_size, var_penalty,
                 activation='linear', dropout=0., adaptive=False,
                 init='normal'):
        super().__init__()

        self.init = init

        self.linear = DropoutLinear(in_features, latent_size,
                                    adaptive=adaptive,
                                    var_penalty=var_penalty,
                                    sparsify=False,
                                    init=init,
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
        if isinstance(self.init, str):
            if self.init == 'normal':
                self.linear.reset_parameters()
            elif self.init == 'symmetric':
                super().reset_parameters()
                dataset = fetch_atlas_modl()
                assign = dataset['assign512']
                assign = np.load(assign).tolist()
                self.linear.weight.data += self.linear.weight.data[:, assign]
                self.linear.weight.data /= 2
            elif self.init == 'orthogonal':
                nn.init.orthogonal_(self.linear.weight.data,
                                    gain=1 / math.sqrt(
                                        self.linear.weight.shape[1]))
            elif self.init == 'rest':
                assert self.linear.out_features == 128
                assert self.linear.in_features == 512
                dataset = fetch_atlas_modl()
                weight = np.load(dataset['loadings128'])
                self.linear.weight.data = torch.from_numpy(np.array(weight))
            elif self.init == 'rest_gm':
                assert self.linear.out_features == 128
                assert self.linear.in_features == 453
                dataset = fetch_atlas_modl()
                weight = np.load(dataset['loadings128_gm'])
                self.linear.weight.data = torch.from_numpy(np.array(weight))
            else:
                raise ValueError('Wrong parameter for `init` %s' % self.init)
        elif isinstance(self.init, np.ndarray):
            self.linear.weight.data = torch.from_numpy(self.init)
        else:
            raise ValueError('Wrong parameter for `init` %s' % self.init)
        self.linear.reset_dropout()

    def penalty(self):
        return self.linear.penalty()


class LatentClassifier(nn.Module):
    def __init__(self, latent_size, target_size, var_penalty,
                 dropout=0., adaptive=False, batch_norm=True):
        super().__init__()

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(latent_size, affine=False, )
        self.linear = DropoutLinear(latent_size,
                                    target_size, bias=True, p=dropout,
                                    var_penalty=var_penalty,
                                    adaptive=adaptive,
                                    level='layer')

    def forward(self, input, logits=False):
        if hasattr(self, 'batch_norm'):
            if not self.training or len(input) > 1:
                input = self.batch_norm(input)
        logits_ = self.linear(input)
        if logits:
            return logits_
        else:
            return F.log_softmax(logits_, dim=1)

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
                 init='normal',
                 batch_norm=True,
                 adaptive='embedding+classifier',
                 ):
        super().__init__()

        embedder_adaptive = 'embedding' in adaptive

        total_length = sum(list(lengths.values()))
        self.embedder = Embedder(in_features, latent_size,
                                 dropout=input_dropout,
                                 adaptive=embedder_adaptive,
                                 init=init,
                                 var_penalty=regularization / total_length,
                                 activation=activation)
        classifier_adaptive = 'classifier' in adaptive
        self.classifiers = {study: LatentClassifier(
            latent_size, target_size, dropout=latent_dropout,
            var_penalty=regularization / lengths[study],
            batch_norm=batch_norm,
            adaptive=classifier_adaptive, )
            for study, target_size in target_sizes.items()}
        for study, classifier in self.classifiers.items():
            self.add_module('classifier_%s' % study, classifier)

    def reset_parameters(self):
        self.embedder.reset_parameters()
        for classifier in self.classifiers.values():
            classifier.reset_parameters()

    def forward(self, inputs, logits=False):
        preds = {}
        for study, input in inputs.items():
            preds[study] = self.classifiers[study](self.embedder(input),
                                                   logits=logits)
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


class FactoredClassifier(BaseEstimator):
    def __init__(self, latent_size=30,
                 activation='linear',
                 batch_size=128, optimizer='sgd',
                 lr=0.001,
                 dropout=0.5, input_dropout=0.25,
                 max_iter=10000, verbose=0,
                 device=-1,
                 regularization=1.,
                 weight_power=0.5,
                 target_study=None,
                 variational=False,
                 batch_norm=True,
                 sampling='cycle',
                 epoch_counting='all',
                 init='normal',
                 refit_from=None,
                 refit_data=['dropout', 'classifier'],
                 adaptive_dropout=True,
                 n_jobs=1,
                 patience=200,
                 seed=None):

        self.latent_size = latent_size
        self.activation = activation
        self.input_dropout = input_dropout
        self.dropout = dropout

        self.regularization = regularization

        self.refit_from = refit_from

        self.sampling = sampling
        self.batch_size = batch_size

        self.batch_norm = batch_norm

        self.optimizer = optimizer
        self.lr = lr

        self.weight_power = weight_power

        self.variational = variational

        self.patience = patience
        self.max_iter = max_iter

        self.target_study = target_study

        self.adaptive_dropout = adaptive_dropout

        self.epoch_counting = epoch_counting

        self.verbose = verbose
        self.device = device
        self.seed = seed

        self.refit_data = refit_data

        self.init = init
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
            adaptive='',
            init=self.init,
            batch_norm=self.batch_norm,
            regularization=self.regularization,
            activation=self.activation,
            lengths=eff_lengths,
            latent_size=self.latent_size,
            target_sizes=target_sizes)

        if self.refit_from is not None:
            (latent_coefs, classif_coefs, classif_biases,
             dropout) = load(self.refit_from)
            module.embedder.linear.weight.data = torch.from_numpy(latent_coefs)
            module.embedder.linear.bias.data.fill_(0.)
            for study, classifier in module.classifiers.items():
                if 'classifier' in self.refit_data:
                    coefs = torch.from_numpy(classif_coefs[study])
                    biases = torch.from_numpy(classif_biases[study])
                    if hasattr(classifier, 'batch_norm'):
                        with torch.no_grad():
                            embedding = module.embedder(X[study])
                            mean = embedding.mean(dim=0)
                            embedding = embedding - mean[None, :]
                            var = embedding.var(dim=0)
                            denom = (torch.sqrt(var) + 1e-5)
                            biases += torch.matmul(coefs, mean)
                            coefs *= denom[None, :]
                            classifier.batch_norm.running_var = var
                            classifier.batch_norm.running_mean = mean
                            classifier.batch_norm.momemtum = 0
                    classifier.linear.weight.data = coefs
                    classifier.linear.bias.data = biases
                if 'dropout' in self.refit_data:
                    p = dropout[study]
                    log_alpha = np.log(p / (1 - p))
                    classifier.linear.log_alpha.fill_(log_alpha)

        # Verbosity + epoch counting
        if self.epoch_counting == 'target_study':
            if self.target_study is None:
                raise ValueError('`target_study` should be specified'
                                 ' if `epoch_counting` is "target_study".')
            else:
                n_samples = ceil(len(X[self.target_study]) / study_weights[
                    self.target_study])
        elif self.epoch_counting == 'all':
            n_samples = sum(len(this_X) for this_X in X.values())
        else:
            raise ValueError('Wrong value for `epoch_counting`: %s' %
                             self.epoch_counting)

        epoch = 0
        for phase in ['pretrain', 'train', 'sparsify']:
            if self.max_iter[phase] == 0:
                continue
            if self.verbose != 0:
                report_every = ceil(self.max_iter[phase] / self.verbose)
            else:
                report_every = None

            print('Phase :', phase)
            print('------------------------------')
            if phase == 'pretrain':
                module.embedder.linear.weight.requires_grad = False
                module.embedder.linear.bias.requires_grad = False

                optimizer = Adam(filter(lambda p: p.requires_grad,
                                        module.parameters()),
                                 lr=self.lr[phase], amsgrad=True)
            elif phase == 'train':
                module.embedder.linear.weight.requires_grad = True
                module.embedder.linear.bias.requires_grad = True
                if self.adaptive_dropout:
                    for classifier in module.classifiers.values():
                        classifier.linear.make_adaptive()
                optimizer = Adam(filter(lambda p: p.requires_grad,
                                        module.parameters()),
                                 lr=self.lr[phase], amsgrad=True)
            else:
                module.embedder.linear.make_additive()
                optimizer = Adam(filter(lambda p: p.requires_grad,
                                        module.parameters()),
                                 lr=self.lr[phase], amsgrad=True)

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
                    if (report_every is not None
                            and epoch % report_every == 0):
                        density = module.embedder.linear.density
                        print('Epoch %.2f, train loss: %.4f, penalty: %.4f,'
                              ' density: %.4f'
                              % (epoch, epoch_loss, epoch_penalty, density))
                        dropout = {}
                        for study, classifier in self.module_.classifiers.items():
                            dropout[study] = classifier.linear.get_p().item()
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
                        print('-----------------------------------')
                        self.dropout_ = dropout

                        break
                batch_size = sum(input.shape[0] for input in inputs.values())
                seen_samples += batch_size
                optimizer.zero_grad()
                module.train()
                preds = module(inputs)
                loss = loss_function(preds, targets)
                penalty = module.penalty(inputs)
                loss += penalty
                loss.backward()
                optimizer.step()

                epoch_batch += 1
                epoch_loss *= (1 - 1 / epoch_batch)
                epoch_loss += loss.item() / epoch_batch

                epoch_penalty *= (1 - 1 / epoch_batch)
                epoch_penalty += penalty.item() / epoch_batch

                epoch = floor(seen_samples / n_samples)
        print('Final density %s' % module.embedder.linear.density)
        phase = 'finetune'
        if self.max_iter[phase] > 0:
            if self.verbose != 0:
                report_every = ceil(self.max_iter[phase] / self.verbose)
            else:
                report_every = None
            X_red = {}

            if self.target_study is not None:
                X = {self.target_study: X[self.target_study]}
            for study, this_X in X.items():
                print('Fine tuning %s' % study)
                this_X = this_X.to(device=device)
                with torch.no_grad():
                    self.module_.embedder.eval()
                    X_red[study] = self.module_.embedder(this_X).to(
                        device=device)
                data = TensorDataset(X_red[study], y[study])
                data_loader = DataLoader(data, shuffle=True,
                                         batch_size=self.batch_size,
                                         drop_last=False,
                                         pin_memory=device.type == 'cuda')
                this_module = module.classifiers[study]
                this_module.reset_parameters()
                if this_module.linear.adaptive:
                    this_module.linear.make_non_adaptive()
                optimizer = Adam(filter(lambda p: p.requires_grad,
                                        this_module.parameters()),
                                 lr=self.lr[phase], amsgrad=True)
                # scheduler = CosineAnnealingLR(optimizer, 50)
                loss_function = F.nll_loss

                seen_samples = 0
                best_loss = float('inf')
                no_improvement = 0
                epoch = 0
                best_state = module.state_dict()
                for epoch in range(self.max_iter[phase]):
                    # scheduler.step(epoch)
                    epoch_batch = 0
                    epoch_penalty = 0
                    epoch_loss = 0
                    for input, target in data_loader:
                        batch_size = input.shape[0]
                        input = input.to(device=device)
                        target = target.to(device=device)

                        this_module.train()
                        optimizer.zero_grad()
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
                            'Epoch %.2f, train loss: %.4f,'
                            ' penalty: %.4f, p: %.2f'
                            % (epoch, epoch_loss, epoch_penalty,
                               this_module.linear.get_p().item()))
                        callback(self, epoch)

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
                      ' %.4f, best model loss %.2f' %
                      (epoch, epoch_loss, best_loss))
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
                "available. Loading on CPU instead.")
            state['device'] = 'cpu'

        self.__dict__.update(state)
