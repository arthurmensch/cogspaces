import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from cogspaces.model_selection import train_test_split


class StudySelector(BaseEstimator):
    def __init__(self, classifier, target_study,
                 n_jobs=1, n_splits=3,
                 seed=None,
                 n_runs=10):
        self.classifier = classifier
        self.target_study = target_study

        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.n_runs = n_runs

        self.seed = seed

    def fit(self, X, y, callback=None):
        sources = list(X.keys())
        sources.remove(self.target_study)
        self.studies_ = [self.target_study]

        if len(sources) > 0:
            seeds = check_random_state(
                self.seed).randint(0, np.iinfo('int32').max,
                                   size=self.n_splits)
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(transferability)(
                    self.classifier, X, y, self.target_study, source, seed)
                for seed in seeds
                for source in [None] + sources)
            transfer = []
            scores = iter(scores)
            for seed in seeds:
                baseline_score = next(scores)
                for source in sources:
                    score = next(scores)
                    transfer.append(dict(seed=seed, source=source,
                                         score=score,
                                         diff=score - baseline_score))
            transfer = pd.DataFrame(transfer)
            transfer = transfer.set_index(['source', 'seed'])
            transfer = transfer.groupby('source').agg('mean')
            print('Pair transfers:')
            transfer = transfer.sort_values('diff', ascending=False)
            print(transfer)
            transfer = transfer.query('diff > 0.005')
            positive = transfer.index.get_level_values(
                'source').values.tolist()
            print('Transfering datasets:', positive)
            self.studies_ += positive
        X = {study: X[study] for study in self.studies_}
        y = {study: y[study] for study in self.studies_}
        self.classifier_ = self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier_.predict(X)


def transferability(classifier, X, y, target, source=None, seed=0):
    train_data, val_data, train_targets, val_targets = \
        train_test_split(X, y, random_state=seed)
    y_val_true = val_targets[target]['contrast'].values
    X_val = {target: val_data[target]}
    if source is None:
        train_data = {target: train_data[target]}
        train_targets = {target: train_targets[target]}
    else:
        train_data = {target: train_data[target],
                      source: train_data[source]}
        train_targets = {target: train_targets[target],
                         source: train_targets[source]}
    classifier.fit(train_data, train_targets)
    y_val_pred = classifier.predict(X_val)[target]['contrast'].values
    return accuracy_score(y_val_true, y_val_pred)
