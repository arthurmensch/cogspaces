import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC


class MultiLogisticClassifier(BaseEstimator):
    def __init__(self, l2_penalty=1e-4, verbose=0, max_iter=1000,
                 estimator='logistic',
                 solver='lbfgs', refit_from=None):
        self.l2_penalty = l2_penalty
        self.verbose = verbose
        self.max_iter = max_iter
        self.solver = solver
        self.refit_from = refit_from

        self.estimator = estimator

    def fit(self, X, y, callback=None):
        self.estimators_ = {}

        if self.refit_from is not None:
            (self.proj_, _, _, _) = load(self.refit_from)
            self.sc_ = {}
        for study in X:
            n_samples = X[study].shape[0]
            if self.refit_from is not None:
                this_X = X[study].dot(self.proj_.T)
                # self.sc_[study] = StandardScaler().fit(this_X)
                # this_X = self.sc_[study].transform(this_X)
            else:
                this_X = X[study]
            this_y = y[study]['contrast']
            groups = y[study]['subject']
            C = 1. / (n_samples * np.array(self.l2_penalty))

            cv = GroupShuffleSplit(n_splits=10, test_size=0.5)

            splits = [(train, test) for train, test in cv.split(this_X, this_y, groups)]

            if self.estimator == 'logistic':
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
            else:
                self.estimators_[study] = LinearSVC(
                    multi_class='ovr',
                    dual=this_X.shape[0] < this_X.shape[1],
                    C=1, max_iter=self.max_iter,
                    tol=1e-4,
                    verbose=self.verbose).fit(this_X, this_y)

    def predict_proba(self, X):
        res = {}
        for study, this_X in X:
            if self.refit_from is not None:
                this_X = this_X.dot(self.proj_.T)
                # this_X = self.sc_[study].transform(this_X)
            res[study] = self.estimators_[study].predict_proba(this_X)
        return res

    def predict(self, X):
        res = {}
        for study, this_X in X.items():
            if self.refit_from is not None:
                this_X = this_X.dot(self.proj_.T)
                # this_X = self.sc_[study].transform(this_X)
            res[study] = pd.DataFrame(dict(
                contrast=self.estimators_[study].predict(this_X),
                subject=0, study=0))
        return res

    @property
    def coef_(self):
        if self.refit_from is not None:
            return {study: estimator.coef_.dot(self.proj_) for study, estimator
                    in self.estimators_.items()}
        else:
            return {study: estimator.coef_ for study, estimator
                    in self.estimators_.items()}