import itertools
import math
from math import ceil, floor

import numpy as np
import pandas as pd
import tempfile
import torch
import torch.nn.functional as F
import warnings
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple

from cogspaces.models.factored import Identity


def infinite_iter(iterable):
    while True:
        for elem in iterable:
            yield elem


class OurAdagrad(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
        super(OurAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.zeros_like(p.data)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (
                        1 + (state['step'] - 1) * group['lr_decay'])

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)
            if group['soft_thresholding'] != 0:
                p.data = soft_thresh(p.data,
                                     group['soft_thresholding']
                                     * clr)
        return loss


class OurAdam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 clip='none',
                 soft_thresholding=0, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        soft_thresholding=soft_thresholding,
                        clip=clip,
                        weight_decay=weight_decay)
        super(OurAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, '
                        'please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['clip'] == 'cross':
                        state['cross_count'] = torch.zeros_like(p.data).long()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                if group['clip'] == 'cross':
                    old_sgn = torch.sign(p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['soft_thresholding'] != 0:
                    p.data = soft_thresh(p.data,
                                         group['soft_thresholding']
                                         * step_size)

                if group['clip'] == 'cross':
                    sgn = torch.sign(p.data)
                    state['cross_count'] += (old_sgn * sgn == -1).long()
                    p.data[state['cross_count'] > 1] = 0
                elif group['clip'] == 'positive':
                    p.data[p.data < 0] = 0.

        return loss


class Embedder(nn.Module):
    def __init__(self, in_features, latent_size, input_dropout=0.,
                 regularization=0.,
                 activation='linear'):
        super().__init__()

        self.input_dropout = nn.Dropout(p=input_dropout)
        self.linear = nn.Linear(in_features, latent_size, bias=True)
        self.regularization = regularization

        if activation == 'linear':
            self.activation = Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, input):
        return self.activation(self.linear(self.input_dropout(input)))

    def reset_parameters(self):
        self.linear.reset_parameters()

    def penalty(self):
        alpha = 1000
        return torch.sum((F.softplus(-alpha * self.linear.weight)
                         + F.softplus(alpha * self.linear.weight)) / alpha)


class LatentClassifier(nn.Module):
    def __init__(self, latent_size, target_size, dropout=0.,
                 ):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(latent_size)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(latent_size, target_size, bias=True)

    def forward(self, input):
        input = self.batch_norm(input)
        input = self.dropout(input)
        return F.log_softmax(self.linear(input), dim=1)

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.batch_norm.reset_parameters()

    def penalty(self):
        return torch.sum(torch.abs(self.linear.weight))
        # return Variable(torch.FloatTensor([0]))


class MultiStudyModule(nn.Module):
    def __init__(self, in_features,
                 latent_size,
                 target_sizes,
                 input_dropout=0.,
                 dropout=0.,
                 regularization=0.,
                 activation='linear'):
        super().__init__()

        self.embedder = Embedder(in_features, latent_size,
                                 input_dropout=input_dropout,
                                 activation=activation,
                                 regularization=regularization,
                                 )

        self.classifiers = {study: LatentClassifier(latent_size, target_size,
                                                    dropout=dropout)
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

    def penalty(self):
        return (self.embedder.penalty(),
                {study: classifier.penalty() for study, classifier
                 in self.classifiers.items()})


class MultiStudyLoss(nn.Module):
    def __init__(self, study_weights: Dict[str, float],
                 ) -> None:
        super().__init__()
        self.study_weights = study_weights

    def forward(self, preds: Dict[str, torch.FloatTensor],
                targets: Dict[str, torch.LongTensor]) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        loss = 0
        for study in preds:
            pred = preds[study]
            target = targets[study]
            this_loss = F.nll_loss(pred, target, size_average=True)
            loss += this_loss * self.study_weights[study]
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
                                     pin_memory=loader.device.type == 'cuda')
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

        self.device = loader.device

    def __next__(self):
        inputs, targets = {}, {}
        if self.sampling == 'all':
            for study in self.studies:
                input, target = next(self.loader_iters[study])
                input = input.to(device=self.device)
                target = target.to(device=self.device)
                inputs[study], targets[study] = input, target
        else:
            study = next(self.study_iter)
            input, target = next(self.loader_iters[study])
            input = input.to(device=self.device)
            target = target.to(device=self.device)
            inputs[study], targets[study] = input, target
        return inputs, targets


class MultiStudyLoader:
    def __init__(self, studies,
                 batch_size=128, sampling='cycle',
                 study_weights=None, seed=None, device=-1):
        self.studies = studies
        self.batch_size = batch_size
        self.sampling = sampling
        self.study_weights = study_weights

        self.device = device
        self.seed = seed

    def __iter__(self):
        return MultiStudyLoaderIter(self)


def soft_thresh(input: torch.Tensor, value: float) -> torch.Tensor:
    sgn = torch.sign(input)
    input *= sgn
    input -= value
    input[input < 0] = 0
    input *= sgn
    return input


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
                 regularization=1e-3,
                 sampling='cycle',
                 patience=2000,
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

        self.regularization = regularization

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
                                       device=device)
        # Model
        target_sizes = {study: int(this_y.max()) + 1
                        for study, this_y in y.items()}
        in_features = next(iter(X.values())).shape[1]

        self.module_ = MultiStudyModule(
            in_features=in_features,
            activation=self.activation,
            latent_size=self.latent_size,
            input_dropout=self.input_dropout,
            regularization=self.regularization,
            dropout=self.dropout,
            target_sizes=target_sizes)
        self.module_.reset_parameters()

        # Loss function
        if study_weights is None or self.sampling == 'random':
            study_weights = {study: 1. for study in X}
        loss_function = MultiStudyLoss(study_weights)
        # Optimizers
        embedder_weight = self.module_.embedder.linear.weight
        params = []
        for name, param in self.module_.named_parameters():
            if name != 'embedder.linear.weight':
                params.append(param)
        if self.optimizer == 'adam':
            optimizer = Adam(self.module_.parameters(), lr=self.lr,
                             amsgrad=True)
            # optimizer = OurAdam([dict(params=params,
            #                           # soft_thresholding=self.regularization
            #                           ),
            #                      dict(params=embedder_weight,
            #                           soft_thresholding=0,
            #                           clip='none')
            #                      ],
            #                     lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = SGD(self.module_.parameters(), lr=self.lr, )
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
        n_elem = self.module_.embedder.linear.weight.view(-1).shape[0]
        indices = check_random_state(10).permutation(n_elem)[
                          :100].tolist()
        self.recorded_ = []
        for inputs, targets in data_loader:
            batch_size = sum(input.shape[0] for input in inputs.values())
            seen_samples += batch_size
            self.module_.train()
            optimizer.zero_grad()

            preds = self.module_(inputs)
            # weights.append(self.module_.embedder.linear.weight.data.view(-1)[random_features].numpy())
            embedder_penalty, penalties = self.module_.penalty()
            # penalty = self.regularization * (sum(penalties.values()))
            penalty = self.regularization * embedder_penalty
            loss = loss_function(preds, targets)
            # loss += penalty
            loss.backward()
            optimizer.step()

            epoch_batch += 1
            epoch_loss *= (1 - 1 / epoch_batch)
            epoch_loss += loss.data[0] / epoch_batch

            epoch = floor(seen_samples / n_samples)
            if epoch > old_epoch:
                self.recorded_.append(self.module_.embedder.linear.weight
                                      .detach().view((-1))[
                                          indices].numpy())
                old_epoch = epoch
                epoch_batch = 0

                if (report_every is not None
                        and epoch % report_every == 0):
                    weight = self.module_.embedder.linear.weight
                    density = (weight != 0).float().mean().data[0]
                    print('Epoch %.2f, train loss: %.4f, penalty: %.4f,'
                          ' density: %.4f' % (epoch, epoch_loss,
                                              penalty, density))
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
                    callback(self, epoch)
                    print('----------------------------------')
                    break
        self.recorded_ = np.concatenate([record[None, :] for
                                         record in self.recorded_], axis=0)
        X_red = {}
        self.module_.embedder.input_dropout.p = 0
        for study, this_X in X.items():
            print('Fine tuning %s' % study)
            this_X = this_X.to(device=device)
            with torch.no_grad():
                X_red[study] = self.module_.embedder(this_X).cpu()
            data = TensorDataset(X_red[study], y[study])
            data_loader = DataLoader(data, shuffle=True,
                                     batch_size=self.batch_size,
                                     pin_memory=cuda)
            module = self.module_.classifiers[study]
            optimizer = Adam(module.parameters(), lr=self.lr * .1)
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
                    input = input.to(device=device)
                    target = target.to(device=device)

                    module.train()
                    # module.batch_norm.eval()
                    optimizer.zero_grad()
                    pred = module(input)
                    loss = loss_function(pred, target)
                    loss.backward()
                    optimizer.step()

                    seen_samples += batch_size
                    epoch_batch += 1
                    epoch_loss *= (1 - 1 / epoch_batch)
                    epoch_loss = loss.data[0] / epoch_batch

                if (report_every is not None
                        and epoch % report_every == 0):
                    print('Epoch %.2f, train loss: %.4f'
                          % (epoch, epoch_loss))

                if epoch_loss > best_loss:
                    no_improvement += 1
                else:
                    no_improvement = 0
                    best_loss = epoch_loss
                    best_state = module.state_dict()

                if no_improvement > self.patience:
                    break
            print('Stopping at epoch %.2f, best train loss'
                  ' %.4f' % (epoch, best_loss))
            module.load_state_dict(best_state)
            print('----------------------------------')
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
        device = self._chec()
        self.module_.eval()
        X_ = {}
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            this_X = this_X.to(device=device)
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
