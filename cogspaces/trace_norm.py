from math import sqrt

from numpy.linalg import svd
from sklearn.base import BaseEstimator

import numpy as np
from sklearn.metrics import log_loss
from sklearn.utils import check_random_state


def proximal_operator(coef, threshold):
    U, s, V = svd(coef, full_matrices=False)
    s = np.maximum(s - threshold, 0)
    rank = np.sum(s != 0)
    U *= s
    return np.dot(U, V), rank


def trace_norm(coef):
    _, s, _ = svd(coef, full_matrices=False)
    return np.sum(s)


def lipschitz_constant(Xs, fit_intercept=False):
    max_squared_sums = np.array([(X ** 2).sum(axis=1).max() for X in Xs])
    max_squared_sums = np.mean(max_squared_sums)
    L = (0.5 * (max_squared_sums + int(fit_intercept)))
    return L

def quad_approx()


class TraceNormEstimator(BaseEstimator):
    def __init__(self, alpha=1, n_iter=1000,
                 momentum=True,
                 fit_intercept=True,
                 verbose=False,
                 random_state=None):
        self.alpha = alpha
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.momentum = momentum
        self.verbose = verbose
        self.random_state = random_state

    @property
    def coefs_(self):
        return [self.coef_[:, this_slice] for this_slice in self.slices_]

    @property
    def intercepts_(self):
        return [self.intercept_[this_slice] for this_slice in self.slices_]

    def fit(self, Xs, ys):
        n_datasets = len(Xs)
        n_features = Xs[0].shape[1]

        sizes = np.array([this_y.shape[1] for this_y in ys])
        limits = [0] + np.cumsum(sizes).tolist()
        total_size = limits[-1]

        self.slices_ = []
        for iter in range(n_datasets):
            self.slices_.append(slice(limits[iter], limits[iter + 1]))

        self.random_state = check_random_state(self.random_state)
        self.coef_ = self.random_state.standard_normal(
            (n_features, total_size))
        self.intercept_ = np.zeros(total_size)

        L = lipschitz_constant(Xs, self.fit_intercept)

        coefs = self.coefs_
        intercepts = self.intercepts_
        if self.momentum:
            t = 1
            prox_coef = np.zeros((n_features, total_size))
            old_prox_coef = np.zeros((n_features, total_size))

        for iter in range(self.n_iter):
            preds = self.predict(Xs)
            for X, y, pred, coef, intercept in zip(Xs, ys, preds,
                                                   coefs, intercepts):
                coef_grad = X.T.dot(y - pred) / X.shape[0] / n_datasets
                intercept_grad = (y - pred).mean(axis=0) / n_datasets

                coef += coef_grad / L
                intercept += intercept_grad / L

            prox_coef[:], rank = proximal_operator(self.coef_, self.alpha / L)

            if self.momentum:
                old_t = t
                t = .5 * (1 + sqrt(1 + 4 * old_t ** 2))
                # Write inplace so that coefs stays valid
                self.coef_[:] = prox_coef * (1 + (old_t - 1) / t)
                self.coef_ -= (old_t - 1) / t * old_prox_coef
                old_prox_coef[:] = prox_coef
            else:
                self.coef_[:] = prox_coef

    def predict(self, Xs):
        coefs = self.coefs_
        intercepts = self.intercepts_
        preds = []
        for X, coef, intercept in zip(Xs, coefs, intercepts):
            logits = X.dot(coef) + intercept
            logits -= logits.max()
            pred = np.exp(logits)
            pred /= np.sum(pred, axis=1, keepdims=True)
            preds.append(pred)
        return preds

    def score(self, Xs, ys):
        preds = self.predict(Xs)
        scores = []
        for pred, y in zip(preds, ys):
            scores.append(log_loss(y, pred))
        return scores
