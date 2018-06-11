import copy
import numpy as np
from joblib import Parallel, delayed
from modl import DictFact
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from cogspaces.datasets.dictionaries import fetch_atlas_modl


def explained_variance(X, components, per_component=True):
    """Score function based on explained variance

        Parameters
        ----------
        data: ndarray,
            Holds single subject data to be tested against components

        per_component: boolean,
            Specify whether the explained variance ratio is desired for each
            map or for the global set of components_

        Returns
        -------
        score: ndarray,
            Holds the score for each subjects. score is two dimensional if
            per_component = True
        """
    full_var = np.var(X)
    n_components = components.shape[0]
    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components = components / S[:, np.newaxis]
    projected_data = X.dot(components.T)
    if per_component:
        res_var = np.zeros(n_components)
        for i in range(n_components):
            res = X - projected_data[:, i][:, None] * components[i][None, :]
            res_var[i] = np.var(res)
        return np.maximum(0., 1. - res_var / full_var)
    else:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(components.T, X.T)
        residuals = X - lr.coef_.dot(components)
        res_var = np.var(residuals)
        return np.maximum(0., 1. - res_var / full_var)



class FactoredDL(BaseEstimator):
    def __init__(self, classifier, n_jobs=1, seed=None, n_runs=2):
        self.classifier = classifier

        self.n_jobs = n_jobs
        self.n_runs = n_runs

        self.seed = seed

    def fit(self, X, y, callback=None):
        loadings_128 = fetch_atlas_modl()['loadings128']
        dict_init = np.load(loadings_128)

        self.classifier_ = copy.deepcopy(self.classifier)

        self.classifier_.n_jobs = 1
        self.classifier_.init = dict_init

        seeds = check_random_state(
            self.seed).randint(0, np.iinfo('int32').max,
                               size=self.n_runs)
        components = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(compute_coefs)(
                self.classifier_, X, y, seed)
            for seed in seeds)

        coefs = np.concatenate(components, axis=0)
        sc = StandardScaler(with_std=False, with_mean=True)
        sc.fit(coefs)
        coefs_ = sc.transform(coefs)

        dict_fact = DictFact(comp_l1_ratio=0, comp_pos=False,
                             n_components=128,
                             code_l1_ratio=0, batch_size=32,
                             learning_rate=1,
                             dict_init=dict_init,
                             code_alpha=5e-5, verbose=0, n_epochs=3,
                             )
        dict_fact.fit(coefs_)
        dict_init = dict_fact.components_
        dict_fact = DictFact(comp_l1_ratio=1, comp_pos=False, n_components=128,
                             code_l1_ratio=0, batch_size=32, learning_rate=1,
                             dict_init=dict_init,
                             code_alpha=5e-5, verbose=10, n_epochs=40)
        dict_fact.fit(coefs_)
        components = dict_fact.components_

        exp_vars = explained_variance(coefs[:500], components,
                                      per_component=True)
        sort = np.argsort(exp_vars)[::-1]

        self.components_dl_ = components[sort]

        self.classifier_.init = self.components_dl_
        self.classifier_.lr['train'] = self.classifier.lr['train'] / 10
        self.classifier_.max_iter = self.classifier.max_iter
        self.classifier_.n_jobs = self.classifier.n_jobs

        self.classifier_.fit(X, y)
        self.components_ = self.classifier_.module_.embedder.linear.weight.detach().numpy()
        return self

    def predict(self, X):
        return self.classifier_.predict(X)


def compute_coefs(classifier, X, y, seed=0):
    classifier.seed = seed
    classifier.fit(X, y)
    return classifier.module_.embedder.linear.weight.detach().numpy()
