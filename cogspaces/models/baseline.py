import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
            self.sc_ = {}
        for study in X:
            n_samples = X[study].shape[0]
            X[study] = X[study].dot(self.proj_.T)
            self.sc_[study] = StandardScaler().fit(X[study])
            X_red = self.sc_[study].transform(X[study])
            C = 1. / (n_samples * self.l2_penalty)
            self.estimators_[study] = LogisticRegression(
                solver=self.solver,
                multi_class='multinomial',
                C=C, max_iter=self.max_iter,
                tol=0,
                verbose=self.verbose).fit(X_red, y[study]['contrast'])

    def predict_proba(self, X):
        res = {}
        for study, this_X in X:
            X_red = self.sc_[study].transform(this_X.dot(self.proj_.T))
            res[study] = self.estimators_[study].predict_proba(X_red)
        return res

    def predict(self, X):
        res = {}
        for study, this_X in X.items():
            X_red = self.sc_[study].transform(this_X.dot(self.proj_.T))
            res[study] = pd.DataFrame(dict(
                contrast=self.estimators_[study].predict(X_red),
                subject=0, study=0))
        return res

    @property
    def coef_(self):
        return {study: estimator.coef_.dot(self.proj_) for study, estimator
                in self.estimators_.items()}
