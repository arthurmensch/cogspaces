from math import sqrt, log

import numpy as np
from numba import jit
from numpy.linalg import svd
from sklearn.base import BaseEstimator


def lipschitz_constant(Xs, fit_intercept=False):
    max_squared_sums = np.array([(X ** 2).sum(axis=1).max() for X in Xs])
    max_squared_sums = np.mean(max_squared_sums)
    L = (0.5 * (max_squared_sums + int(fit_intercept)))
    return L


@jit(nopython=True, cache=False)
def trace_norm(coef):
    _, s, _ = svd(coef, full_matrices=False)
    return np.sum(s)


@jit(nopython=True, cache=True)
def proximal_operator(coef, threshold):
    U, s, V = svd(coef, full_matrices=False)
    s = np.maximum(s - threshold, 0)
    rank = np.sum(s != 0)
    U *= s
    return np.dot(U, V), rank


@jit(nopython=True, cache=True)
def _prox_grad(Xs, ys, preds, coef, intercept,
               prox_coef, prox_intercept, coef_grad, intercept_grad,
               slices, L, alpha, beta):
    n_datasets = len(slices)
    _predict(Xs, preds, coef, intercept, slices)
    loss = 0
    for X, y, pred, this_slice in zip(Xs, ys, preds, slices):
        coef_grad[:, this_slice[0]:this_slice[1]] = np.dot(X.T, pred - y) / \
                                                    X.shape[0] / n_datasets
        for jj, j in enumerate(range(this_slice[0], this_slice[1])):
            intercept_grad[j] = (pred[:, jj] - y[:, jj]).mean() / n_datasets
        loss += cross_entropy(y, pred) / n_datasets
    if beta > 0:
        prox_coef[:] = (1 - beta / L) * coef - coef_grad / L
    else:
        prox_coef[:] = coef - coef_grad / L
    prox_intercept[:] = intercept - intercept_grad / L
    if alpha > 0:
        prox_coef[:], rank = proximal_operator(prox_coef, alpha / L)
    else:
        rank = prox_coef.shape[1]
    return loss, rank


@jit(nopython=True, cache=True)
def _predict(Xs, preds, coef, intercept, slices):
    for X, pred, this_slice in zip(Xs, preds, slices):
        logits = np.dot(X, coef[:, this_slice[0]:this_slice[1]])
        logits += intercept[this_slice[0]:this_slice[1]]
        logits -= logits.max()
        pred[:] = np.exp(logits)
        for i in range(pred.shape[0]):
            pred[i] /= np.sum(pred[i])


@jit(nopython=True, cache=True)
def _loss(Xs, ys, preds, coef, intercept, slices):
    n_datasets = len(slices)
    _predict(Xs, preds, coef, intercept, slices)
    loss = 0
    for y, pred in zip(ys, preds):
        loss += cross_entropy(y, pred) / n_datasets
    return loss


@jit(nopython=True, cache=True)
def _quad_approx(coef, intercept,
                 prox_coef, prox_intercept, coef_grad,
                 intercept_grad,
                 coef_diff, intercept_diff,
                 loss, L):
    approx = loss
    coef_diff[:] = prox_coef - coef
    intercept_diff[:] = prox_intercept - intercept
    approx += np.sum(coef_diff * coef_grad)
    approx += np.sum(intercept_diff * intercept_grad)
    approx += .5 * L * (np.sum(coef_diff ** 2) + np.sum(intercept_diff ** 2))
    return approx


@jit("float32(int64[:, :], float32[:, :])", nopython=True)
def cross_entropy(y_true, y_pred):
    n_samples, n_targets = y_true.shape
    loss = 0
    for i in range(n_samples):
        for j in range(n_targets):
            if y_true[i, j]:
                loss -= log(y_pred[i, j])
    return loss / n_samples


@jit(nopython=True, cache=True)
def _ista_loop(L, Xs, coef, coef_diff, coef_grad, intercept,
               intercept_diff, intercept_grad, old_prox_coef, preds,
               prox_coef, prox_intercept, ys,
               n_iter, max_backtracking_iter, slices, alpha, beta,
               backtracking_frequency, backtracking_divider,
               verbose, momentum):
    old_prox_coef[:] = 0
    t = 1
    _predict(Xs, preds, coef, intercept, slices)
    new_loss = _loss(Xs, ys, preds, coef, intercept, slices)
    rank = np.linalg.matrix_rank(coef)
    for iter in range(n_iter):
        if iter % (n_iter // verbose) == 0:
            penalized_loss = new_loss
            if alpha > 0:
                penalized_loss += alpha * trace_norm(coef)
            if beta > 0:
                penalized_loss += beta / 2 * np.sum(coef ** 2)
            print('Iteration', iter, 'rank', rank, 'loss', penalized_loss)

        for j in range(max_backtracking_iter):
            loss, rank = _prox_grad(Xs, ys, preds, coef, intercept,
                                    prox_coef, prox_intercept, coef_grad,
                                    intercept_grad,
                                    slices, L, alpha, beta)
            if iter % backtracking_frequency != 0:
                break
            new_loss = _loss(Xs, ys, preds, prox_coef, prox_intercept,
                             slices)
            quad_approx = _quad_approx(coef, intercept,
                                       prox_coef, prox_intercept,
                                       coef_grad, intercept_grad,
                                       coef_diff, intercept_diff,
                                       loss, L)
            if new_loss <= quad_approx:
                break
            else:
                if verbose:
                    print('Backtracking step size.')
                L *= backtracking_divider

        if momentum:
            old_t = t
            t = .5 * (1 + sqrt(1 + 4 * old_t ** 2))
            # Write inplace so that coefs stays valid
            coef[:] = prox_coef * (1 + (old_t - 1) / t)
            coef -= (old_t - 1) / t * old_prox_coef
            old_prox_coef[:] = prox_coef
        else:
            coef[:] = prox_coef


class TraceNormEstimator(BaseEstimator):
    def __init__(self, alpha=1., beta=0., n_iter=1000,
                 momentum=True,
                 fit_intercept=True,
                 verbose=False,
                 max_backtracking_iter=10,
                 backtracking_frequency=10,
                 step_size_multiplier=500,
                 backtracking_divider=1.1):
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.momentum = momentum
        self.verbose = verbose

        self.backtracking_frequency = backtracking_frequency
        self.backtracking_divider = backtracking_divider
        self.max_backtracking_iter = max_backtracking_iter
        self.init_multiplier = step_size_multiplier

    def fit(self, Xs, ys):
        n_datasets = len(Xs)
        n_features = Xs[0].shape[1]

        sizes = np.array([this_y.shape[1] for this_y in ys])
        limits = [0] + np.cumsum(sizes).tolist()
        total_size = limits[-1]

        self.slices_ = []
        for iter in range(n_datasets):
            self.slices_.append(np.array([limits[iter], limits[iter + 1]]))
        self.slices_ = tuple(self.slices_)
        L = lipschitz_constant(Xs, self.fit_intercept) / self.init_multiplier
        coef = np.zeros((n_features, total_size), dtype=np.float32)
        intercept = np.zeros(total_size, dtype=np.float32)

        prox_coef = np.empty_like(coef, dtype=np.float32)
        coef_grad = np.empty_like(coef, dtype=np.float32)
        prox_intercept = np.empty_like(intercept, dtype=np.float32)
        intercept_grad = np.empty_like(intercept, dtype=np.float32)
        coef_diff = np.empty_like(coef, dtype=np.float32)
        intercept_diff = np.empty_like(intercept, dtype=np.float32)
        old_prox_coef = np.empty_like(coef)

        preds = tuple(np.empty_like(y, dtype=np.float32) for y in ys)

        _ista_loop(L, Xs, coef, coef_diff, coef_grad, intercept,
                   intercept_diff, intercept_grad, old_prox_coef,
                   preds, prox_coef, prox_intercept, ys,
                   self.n_iter, self.max_backtracking_iter, self.slices_,
                   self.alpha, self.beta,
                   self.backtracking_frequency, self.backtracking_divider,
                   self.verbose, self.momentum
                   )
        self.coef_ = coef
        self.intercept_ = intercept

    def score(self, Xs, ys):
        preds = self.predict(Xs)
        scores = []
        for pred, y in zip(preds, ys):
            scores.append(cross_entropy(y, pred))
        return scores

    def predict(self, Xs):
        preds = tuple(np.empty((X.shape[0], this_slice[1] - this_slice[0]))
                      for X, this_slice in zip(Xs, self.slices_))
        _predict(Xs, preds, self.coef_, self.intercept_, self.slices_)
        return preds
