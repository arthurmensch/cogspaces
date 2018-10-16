import copy

import numpy as np
import torch
from joblib import Parallel, delayed, Memory
from modl import DictFact
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


class EnsembleClassifier(BaseEstimator):
    def __init__(self, estimator, n_jobs=1, seed=None, n_runs=2,
                 alpha=1e-4, memory=Memory(cachedir=None),
                 warmup=True):
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.n_runs = n_runs
        self.alpha = alpha

        self.seed = seed

        self.warmup = warmup

        self.memory = memory

    def fit(self, X, y, callback=None):
        self.estimator_ = copy.deepcopy(self.estimator)
        self.estimator_.max_iter = {'pretrain': 0, 'train': 0, 'finetune': 0}
        self.estimator_.fit(X, y)
        module = self.estimator_.module_

        seeds = check_random_state(self.seed).randint(0, np.iinfo('int32').max,
                                                      size=(self.n_runs, ))
        res = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.memory.cache(_compute_coefs))(
                self.estimator, X, y, seed) for seed in seeds)
        embedder_weights, full_coefs, full_biases = zip(*res)
        embedder_weights = np.concatenate(
            [embedder_weight.numpy() for embedder_weight in embedder_weights],
            axis=0)
        mean_coefs = {
            study: np.mean(np.concatenate([full_coef[study].numpy()[:, :, None]
                                           for full_coef in full_coefs],
                                          axis=2),
                           axis=2)
            for study in full_coefs[0]}
        mean_biases = {
            study: np.mean(np.concatenate([full_bias[study].numpy()[:, None]
                                           for full_bias in
                                           full_biases],
                                          axis=0),
                           axis=0)
            for study in full_coefs[0]}
        embedder_init = module.get_embedder_init().numpy()
        embedder_weight = self.memory.cache(_compute_components)(
            embedder_weights,
            embedder_init,
            self.alpha,
            self.warmup)
        classifiers_weights = {
            study: np.linalg.lstsq(embedder_weight.T, mean_coef, rcond=None)[0].T
            for study, mean_coef in mean_coefs.items()}

        module.embedder.weight.data = torch.from_numpy(embedder_weight)
        module.embedder.bias.data.fill_(0.)
        for study, classifier in module.classifiers.items():
            classifier.linear.weight.data = torch.from_numpy(
                classifiers_weights[study])
            classifier.linear.bias.data = torch.from_numpy(mean_biases[study])
            if hasattr(classifier, 'batch_norm'):
                classifier.batch_norm.running_mean.fill_(0.)
                classifier.batch_norm.running_var.fill_(1.)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def _compute_coefs(estimator, X, y, seed=0):
    estimator.n_jobs = 1
    estimator.seed = seed
    estimator.fit(X, y)

    module = estimator.module_
    weight = estimator.module_.embedder.weight.data
    studies = list(estimator.module_.classifiers.keys())
    in_features = estimator.module_.embedder.in_features
    with torch.no_grad():
        full_bias = module({study: torch.zeros((1, in_features))
                            for study in studies}, logits=True)
        full_coef = module({study: torch.eye(in_features)
                            for study in studies}, logits=True)
        full_coef = {study: full_coef[study] - full_bias[study]
                     for study in studies}
    return weight, full_coef, full_bias


def _compute_components(embedder_weights, embedder_init, alpha, warmup):
    if warmup:
        dict_fact = DictFact(comp_l1_ratio=0, comp_pos=True,
                             n_components=embedder_init.shape[0],
                             code_l1_ratio=0, batch_size=32,
                             learning_rate=1,
                             dict_init=embedder_init,
                             code_alpha=alpha, verbose=0, n_epochs=2,
                             )
        dict_fact.fit(embedder_weights)
        embedder_init = dict_fact.components_
    dict_fact = DictFact(comp_l1_ratio=1, comp_pos=True,
                         n_components=embedder_init.shape[0],
                         code_l1_ratio=0, batch_size=32, learning_rate=1,
                         dict_init=embedder_init,
                         code_alpha=alpha, verbose=10, n_epochs=20)
    dict_fact.fit(embedder_weights)
    components = dict_fact.components_
    return components
