import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class MultiLogisticClassifier(BaseEstimator):
    def __init__(self, l2_penalty=1e-4, verbose=0, max_iter=1000,
                 solver='lbfgs', reduction=None):
        self.l2_penalty = l2_penalty
        self.verbose = verbose
        self.max_iter = max_iter
        self.solver = solver
        self.reduction = reduction

    def fit(self, X, y, callback=None):
        self.estimators_ = {}

        if self.reduction is not None:
            self.proj_ = np.load(self.reduction)

        for study in X:
            n_samples = X[study].shape[0]
            X[study] = X[study].dot(self.proj_.T)
            C = 1. / (n_samples * self.l2_penalty)
            self.estimators_[study] = LogisticRegression(
                solver=self.solver,
                multi_class='multinomial',
                C=C, max_iter=self.max_iter,
                tol=0,
                verbose=self.verbose).fit(X[study], y[study]['contrast'])

    def predict_proba(self, X):
        res = {}
        for study, this_X in X:
            res[study] = self.estimators_[study].predict_proba(this_X.dot(self.proj_.T))
        return res

    def predict(self, X):
        res = {}
        for study, this_X in X.items():
            res[study] = pd.DataFrame(dict(
                contrast=self.estimators_[study].predict(this_X.dot(self.proj_.T)),
                subject=0, study=0))
        return res

    @property
    def coef_(self):
        return {study: estimator.coef_.dot(self.proj_) for study, estimator
                in self.estimators_.items()}
