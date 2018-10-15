import tempfile
from math import ceil, floor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from cogspaces.input_data import MultiStudyLoader
from cogspaces.modules.factored import VarMultiStudyModule
from cogspaces.modules.loss import MultiStudyLoss


class FactoredClassifier(BaseEstimator):
    """

    """
    def __init__(self,
                 latent_size=30,
                 batch_size=128,
                 lr=None,
                 dropout=0.5,
                 input_dropout=0.25,
                 max_iter=None,
                 verbose=0,
                 weight_power=0.5,
                 init='normal',
                 n_jobs=1,
                 patience=200,
                 seed=None):
        """

        :param latent_size:
        :param batch_size:
        :param lr:
        :param dropout:
        :param input_dropout:
        :param max_iter:
        :param verbose:
        :param weight_power:
        :param init:
        :param n_jobs:
        :param patience:
        :param seed:
        """

        if lr is None:
            lr = {'pretrain': 1e-3, 'train': 1e-3, 'finetune': 1e-3}
        if max_iter is None:
            max_iter = {'pretrain': 200, 'train': 300, 'finetune': 200}

        self.latent_size = latent_size
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.weight_power = weight_power

        self.init = init
        self.batch_size = batch_size
        self.lr = lr

        self.patience = patience
        self.max_iter = max_iter

        self.verbose = verbose
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y, callback=None):
        torch.set_num_threads(self.n_jobs)

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
        eff_lengths = {study: total_length * study_weight for
                       study, study_weight
                       in study_weights.items()}

        data_loader = MultiStudyLoader(data, sampling='random',
                                       batch_size=self.batch_size,
                                       seed=self.seed,
                                       study_weights=study_weights,
                                       )
        # Model
        target_sizes = {study: int(this_y.max()) + 1
                        for study, this_y in y.items()}
        in_features = next(iter(X.values())).shape[1]

        if self.latent_size == 'auto':
            latent_size = sum(list(target_sizes.values()))
        else:
            latent_size = self.latent_size

        # Loss
        loss_study_weights = {study: 1. for study in data}
        loss_function = MultiStudyLoss(loss_study_weights, )

        module = self.module_ = VarMultiStudyModule(
            in_features=in_features,
            input_dropout=self.input_dropout,
            latent_dropout=self.dropout,
            adaptive='',
            init=self.init,
            lengths=eff_lengths,
            latent_size=latent_size,
            target_sizes=target_sizes)

        n_samples = sum(len(this_X) for this_X in X.values())

        for phase in ['pretrain', 'train']:
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
            else:  # if phase == 'train':
                module.embedder.linear.weight.requires_grad = True
                module.embedder.linear.bias.requires_grad = True
                for classifier in module.classifiers.values():
                    classifier.linear.make_adaptive()
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
                        print('Epoch %.2f, train loss: %.4f, penalty: %.4f'
                              % (epoch, epoch_loss, epoch_penalty))
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

        phase = 'finetune'
        if self.max_iter[phase] > 0:
            if self.verbose != 0:
                report_every = ceil(self.max_iter[phase] / self.verbose)
            else:
                report_every = None
            X_red = {}

            for study, this_X in X.items():
                print('Fine tuning %s' % study)
                with torch.no_grad():
                    self.module_.embedder.eval()
                    X_red[study] = self.module_.embedder(this_X)
                data = TensorDataset(X_red[study], y[study])
                data_loader = DataLoader(data, shuffle=True,
                                         batch_size=self.batch_size,
                                         drop_last=False,
                                         pin_memory=False)
                this_module = module.classifiers[study]
                if this_module.linear.adaptive:
                    this_module.linear.make_non_adaptive()
                optimizer = Adam(filter(lambda p: p.requires_grad,
                                        this_module.parameters()),
                                 lr=self.lr[phase], amsgrad=True)
                loss_function = F.nll_loss

                seen_samples = 0
                best_loss = float('inf')
                no_improvement = 0
                epoch = 0
                best_state = module.state_dict()
                for epoch in range(self.max_iter[phase]):
                    epoch_batch = 0
                    epoch_penalty = 0
                    epoch_loss = 0
                    for input, target in data_loader:
                        batch_size = input.shape[0]

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
                        print('Epoch %.2f, train loss: %.4f,'
                              ' penalty: %.4f, p: %.2f'
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
                      ' %.4f, best model loss %.2f' %
                      (epoch, epoch_loss, best_loss))
                print('-----------------------------------')

        if callback is not None:
            callback(self, epoch)

        return self

    def predict_log_proba(self, X):
        self.module_.eval()
        X_ = {}
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            X_[study] = this_X
        with torch.no_grad():
            preds = self.module_(X_)
        return {study: pred.data.numpy() for study, pred in
                preds.items()}

    def predict_latent(self, X):
        self.module_.eval()
        latents = {}
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            with torch.no_grad():
                latent = self.module_.embedder(this_X)
            latents[study] = latent.data.numpy()
        return latents

    def predict_rec(self, X):
        self.module_.eval()
        recs = {}
        W = self.module_.embedder.linear.weight
        Ginv = torch.inverse(torch.matmul(W, W.transpose(0, 1)))
        back_proj = torch.matmul(Ginv, W)
        for study, this_X in X.items():
            this_X = torch.from_numpy(this_X).float()
            with torch.no_grad():
                latent = self.module_.embedder(this_X)
                rec = torch.matmul(latent, back_proj)
            recs[study] = rec.numpy()
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
                     study_contrast=0,
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
        for key in ['module_']:
            if key not in state:
                continue
            dump = state.pop(key)
            with tempfile.SpooledTemporaryFile() as f:
                f.write(dump)
                f.seek(0)
                val = torch.load(f)
            state[key] = val

        self.__dict__.update(state)
