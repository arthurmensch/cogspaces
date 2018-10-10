import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


class MultiLogisticClassifier(BaseEstimator):
    def __init__(self, l2_penalty=1e-4, verbose=0, max_iter=1000,
                 estimator='logistic',
                 solver='lbfgs'):
        self.l2_penalty = l2_penalty
        self.verbose = verbose
        self.max_iter = max_iter
        self.solver = solver

        self.estimator = estimator

    def fit(self, X, y, callback=None):
        self.estimators_ = {}

        for study in X:
            n_samples = X[study].shape[0]
            this_X = X[study]
            this_y = y[study]['contrast'].values
            groups = y[study]['subject'].values
            C = 1. / (n_samples * np.array(self.l2_penalty))

            cv = GroupShuffleSplit(n_splits=3, test_size=0.5)

            splits = [(train, test) for train, test in cv.split(this_X, this_y, groups)]

            splits_ = []
            n_classes = this_y.max() + 1
            for train, test in splits:
                if len(np.unique(this_y[train])) == n_classes:
                    splits_.append((train, test))
            splits = splits_

            if len(C) > 1:
                self.estimators_[study] = LogisticRegressionCV(
                    solver=self.solver,
                    cv=splits,
                    multi_class='multinomial',
                    Cs=C, max_iter=self.max_iter,
                    tol=1e-4,
                    verbose=self.verbose).fit(this_X, this_y)
            else:
                self.estimators_[study] = LogisticRegression(
                    solver=self.solver,
                    multi_class='multinomial',
                    C=C[0], max_iter=self.max_iter,
                    tol=1e-4,
                    verbose=self.verbose).fit(this_X, this_y)

    def predict_proba(self, X):
        res = {}
        for study, this_X in X:
            res[study] = self.estimators_[study].predict_proba(this_X)
        return res

    def predict(self, X):
        res = {}
        for study, this_X in X.items():
            res[study] = pd.DataFrame(dict(
                contrast=self.estimators_[study].predict(this_X),
                subject=0, study=0))
        return res

    @property
    def coef_(self):
        return {study: estimator.coef_ for study, estimator
                in self.estimators_.items()}