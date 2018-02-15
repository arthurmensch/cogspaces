import warnings

import numpy as np
import torch
import torch.cuda
import torch.nn.init as init
from sklearn.base import BaseEstimator
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

CUDA = torch.cuda.is_available()
DEVICE = 0


class LatentMultiSoftmax(nn.Module):
    def __init__(self, n_features, n_targets_list, n_components,
                 input_dropout_rate, latent_dropout_rate):
        super().__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_rate)
        self.latent = nn.Linear(n_features, n_components, bias=False)
        self.dropout = nn.Dropout(p=latent_dropout_rate)
        self.classifiers = nn.ModuleList([nn.Linear(n_components,
                                                    n_targets, bias=True)
                                          for n_targets in n_targets_list])
        init.xavier_uniform(self.latent.weight)
        for classifier in self.classifiers:
            init.xavier_uniform(classifier.weight)
            init.uniform(classifier.bias)

    def forward(self, Xs, output_index=None):
        if output_index is not None:
            classifier = self.classifiers[output_index]
            return classifier(self.dropout(
                self.latent(self.input_dropout(Xs))))
        else:
            y_preds = []
            for X, classifier in zip(Xs, self.classifiers):
                y_pred = classifier(self.dropout(
                    self.latent(self.input_dropout(X))))
                y_preds.append(y_pred)
            return y_preds

    def penalty(self):
        penalty = self.latent.weight.norm() ** 2
        for classifier in self.classifiers:
            penalty += classifier.weight.norm() ** 2
        return penalty


class MultiSoftMax(nn.Module):
    def __init__(self, n_features, n_targets_list, input_dropout_rate):
        super().__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_rate)
        self.classifiers = nn.ModuleList([nn.Linear(n_features,
                                                    n_targets, bias=True)
                                          for n_targets in n_targets_list])
        for classifier in self.classifiers:
            init.xavier_uniform(classifier.weight)
            init.uniform(classifier.bias)

    def forward(self, Xs, output_index=None):
        if output_index is not None:
            classifier = self.classifiers[output_index]
            return classifier(self.input_dropout(Xs))
        else:
            y_preds = []
            for X, classifier in zip(Xs, self.classifiers):
                y_pred = classifier(self.input_dropout(X))
                y_preds.append(y_pred)
            return y_preds

    def penalty(self):
        penalty = 0.
        for classifier in self.classifiers:
            penalty += classifier.weight.norm() ** 2
        return penalty


class NonConvexEstimator(BaseEstimator):
    def __init__(self, alpha=1.,
                 n_components=25,
                 step_size=1e-3,
                 latent_dropout_rate=0.,
                 input_dropout_rate=0.,
                 batch_size=256,
                 optimizer='adam',
                 architecture='flat',
                 n_jobs=1,
                 max_iter=1000):
        self.alpha = alpha
        self.n_components = n_components
        self.max_iter = max_iter
        self.step_size = step_size
        self.latent_dropout_rate = latent_dropout_rate
        self.input_dropout_rate = input_dropout_rate
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.optimizer = optimizer
        self.architecture = architecture

    def fit(self, Xs, ys, dataset_weights=None):
        # Input curation
        n_samples = sum(X.shape[0] for X in Xs)
        n_datasets = len(Xs)
        n_features = Xs[0].shape[1]
        n_targets_list = [int(np.max(y)) + 1 for y in ys]

        if self.n_components == 'auto':
            self.n_components = sum(n_targets_list)

        # Data loaders
        Xs = [torch.from_numpy(X) for X in Xs]
        ys = [torch.from_numpy(y) for y in ys]

        datasets = [TensorDataset(X, y) for X, y in zip(Xs, ys)]
        loaders = [DataLoader(dataset, batch_size=self.batch_size,
                              shuffle=True, pin_memory=CUDA,
                              num_workers=0)
                   for dataset in datasets]
        loaders_iter = [iter(loader) for loader in loaders]

        if dataset_weights is None:
            dataset_weights = np.ones(n_datasets, dtype=np.float32)
        else:
            dataset_weights = np.array(dataset_weights, dtype=np.float32)
        dataset_weights /= np.mean(dataset_weights)
        dataset_weights = torch.from_numpy(dataset_weights)
        if CUDA:
            dataset_weights = dataset_weights.cuda(device_id=DEVICE)

        # Model, loss, optimizer
        if self.architecture == 'factored':
            self.model = LatentMultiSoftmax(n_features=n_features,
                                            n_targets_list=n_targets_list,
                                            n_components=self.n_components,
                                            latent_dropout_rate=self.latent_dropout_rate,
                                            input_dropout_rate=self.input_dropout_rate)
        elif self.architecture == 'flat':
            self.model = MultiSoftMax(n_features=n_features,
                                      n_targets_list=n_targets_list,
                                      input_dropout_rate=self.input_dropout_rate)
        criterion = CrossEntropyLoss(size_average=True)

        if CUDA:
            self.model = self.model.cuda(device_id=DEVICE)

        if self.optimizer == 'adam':
            options_list = []
            for name, params in self.model.named_parameters():
                if name.endswith('bias'):
                    # Workaround bug: [params] instead of params
                    # https://discuss.pytorch.org/t/problem-on-different-learning-rate-and-weight-decay-in-different-layers/3619
                    options = {'params': [params],
                               'lr': self.step_size,
                               'weight_decay': 0}
                else:  # name.endswith('weight')
                    options = {'params': [params],
                               'lr': self.step_size,
                               'weight_decay': self.alpha}
                options_list.append(options)
            optimizer = torch.optim.Adam(options_list)

            # Train loop
            n_iter = 0
            old_epoch = -1
            epoch = 0

            while epoch < self.max_iter:
                self.model.train()
                optimizer.zero_grad()
                for i, loader_iter in enumerate(loaders_iter):
                    try:
                        X_batch, y_batch = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loaders[i])
                        X_batch, y_batch = next(loader_iter)
                        loaders_iter[i] = loader_iter
                    batch_len = X_batch.size()[0]
                    X_batch = Variable(X_batch)
                    y_batch = Variable(y_batch)
                    if CUDA:
                        X_batch = X_batch.cuda(device_id=DEVICE)
                        y_batch = y_batch.cuda(device_id=DEVICE)
                    y_pred = self.model(X_batch, output_index=i)
                    loss = criterion(y_pred, y_batch) * dataset_weights[i]
                    loss.backward()
                    # Counting logic
                    n_iter += batch_len
                optimizer.step()

                epoch = n_iter // n_samples
                if epoch > old_epoch:
                    rank = np.linalg.matrix_rank(self.coef_)
                    loss = self._loss(Xs, ys, dataset_weights)
                    print('Epoch %i: train loss %.4f rank %i'
                          % (epoch, loss, rank))
                old_epoch = epoch
        elif self.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(params=self.model.parameters(),
                                          lr=self.step_size)
            self.model.train()
            for epoch in range(self.max_iter):
                def closure():
                    optimizer.zero_grad()
                    total_loss = 0.
                    for i, data in enumerate(datasets):
                        X = Variable(data.data_tensor)
                        y = Variable(data.target_tensor)
                        if CUDA:
                            X_batch = X_batch.cuda(device_id=DEVICE)
                            y_batch = y_batch.cuda(device_id=DEVICE)
                        y_pred = self.model(X, output_index=i)
                        loss = criterion(y_pred, y)
                        loss *= dataset_weights[i]
                        loss.backward()
                        total_loss += loss
                    penalty = self.alpha * .5 * self.model.penalty()
                    penalty.backward()
                    total_loss += penalty
                    return total_loss

                optimizer.step(closure)
                rank = np.linalg.matrix_rank(self.coef_)
                loss = self._loss(Xs, ys, dataset_weights)
                print(
                    'Epoch %i: train loss %.4f rank %i' % (epoch, loss, rank))

    def _loss(self, Xs, ys, dataset_weights, penalty=True):
        criterion = CrossEntropyLoss(size_average=True)
        total_loss = 0.
        self.model.eval()
        for i, (X, y) in enumerate(zip(Xs, ys)):
            X, y = Variable(X), Variable(y)
            if CUDA:
                X, y = X.cuda(device_id=DEVICE), y.cuda(device_id=DEVICE)
            y_pred = self.model(X, output_index=i)
            total_loss += criterion(y_pred, y) * dataset_weights[i]
        if penalty:
            total_loss += self.model.penalty() * .5 * self.alpha
        return total_loss.data[0]

    def predict(self, Xs):
        Xs = [Variable(torch.from_numpy(X)) for X in Xs]
        if CUDA:
            Xs = [X.cuda(device_id=DEVICE) for X in Xs]
        self.model.eval()
        y_preds = self.model(Xs)
        if CUDA:
            y_preds = [y_pred.cpu() for y_pred in y_preds]
        y_preds = [np.argmax(y_pred.data.numpy(), axis=1)
                   for y_pred in y_preds]
        return y_preds


    @property
    def coef_(self):
        if self.architecture == 'factored':
            latent_weight = self.model.latent.weight.data
            classifier_weights = [classifier.weight.data
                                  for classifier in self.model.classifiers]
            coefs = [torch.mm(classifier_weight, latent_weight) for classifier_weight in classifier_weights]
        else:
            coefs = [classifier.weight.data for classifier in self.model.classifiers]
        if CUDA:
            cpu_coefs = []
            for coef in coefs:
                cpu_coef = coef.cpu()
                cpu_coefs.append(cpu_coef)
            coefs = cpu_coefs
        numpy_coefs = []
        for coef in coefs:
            numpy_coefs.append(coef.numpy())
        return numpy_coefs

    @property
    def intercept_(self):
        intercept = [classifier.bias.data
                     for classifier in self.model.classifiers]
        intercept = torch.cat(intercept, dim=0)
        if CUDA:
            intercept = intercept.cpu()
        return intercept.numpy()


class TransferEstimator(NonConvexEstimator):
    def __init__(self, alpha=0.,
                 n_components=25,
                 step_size=1e-3,
                 latent_dropout_rate=0.,
                 input_dropout_rate=0.,
                 batch_size=256,
                 optimizer='adam',
                 architecture='flat',
                 n_jobs=1,
                 max_iter=1000,
                 Xs_helpers=None,
                 ys_helpers=None,
                 dataset_weights_helpers=None):
        super(TransferEstimator,
              self).__init__(alpha=alpha,
                             n_components=n_components,
                             step_size=step_size,
                             latent_dropout_rate=latent_dropout_rate,
                             input_dropout_rate=input_dropout_rate,
                             batch_size=batch_size,
                             optimizer=optimizer,
                             architecture=architecture,
                             n_jobs=n_jobs,
                             max_iter=max_iter)

        self.Xs_helpers = Xs_helpers
        self.ys_helpers = ys_helpers
        self.dataset_weights_helpers = dataset_weights_helpers

    def fit(self, X, y):
        if self.architecture == 'flat':
            if self.Xs_helpers is not None or self.ys_helpers is not None:
                warnings.warn('Ignoring helper datasets'
                              ' as architecture is flat')
            Xs = [X]
            ys = [y]
            dataset_weights = [1.]
        else:
            Xs = [X] + list(self.Xs_helpers)
            ys = [y] + list(self.ys_helpers)
            dataset_weights = [1.] + self.dataset_weights_helpers
        super(TransferEstimator, self).fit(Xs, ys,
                                           dataset_weights=dataset_weights)


    def predict_all(self, Xs):
        return super(TransferEstimator, self).predict(Xs)

    def predict(self, X):
        X = Variable(torch.from_numpy(X))
        if CUDA:
            X = X.cuda(device_id=DEVICE)
        y_pred = self.model(X, output_index=0)

        if CUDA:
            y_pred = y_pred.cpu()
        y_pred = np.argmax(y_pred.data.numpy(), axis=1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    # @property
    # def coef_(self):
    #     if self.architecture == 'factored':
    #         latent_weight = self.model.latent.weight.utils
    #         classifier_weights = self.model.classifiers[0].weight.utils
    #         coef = torch.mm(classifier_weights, latent_weight)
    #     else:
    #         coef = self.model.classifiers[0].weight.utils
    #     if CUDA:
    #         coef = coef.cpu()
    #     return coef.numpy()
    #
    # @property
    # def intercept_(self):
    #     intercept = self.model.classifiers[0].bias.utils
    #     if CUDA:
    #         intercept = intercept.cpu()
    #     return intercept.numpy()
