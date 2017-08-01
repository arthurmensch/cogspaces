import os
from os.path import join

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import load
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder

import numpy as np

from numpy.linalg import pinv, svd

idx = pd.IndexSlice


def get_output_dir(data_dir=None):
    """ Returns the directories in which cogspaces store results.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    paths: list of strings
        Paths of the dataset directories.

    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :
    1. the keyword argument data_dir
    2. the global environment variable OUTPUT_COGSPACES_DIR
    4. output/cogspaces in the user home folder
    """

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        return data_dir
    else:
        # If data_dir has not been specified, then we crawl default locations
        output_dir = os.getenv('OUTPUT_COGSPACES_DIR')
        if output_dir is not None:
            return output_dir
    return os.path.expanduser('~/output/cogspaces')


def make_data_frame(datasets, source,
                    reduced_dir=None, unmask_dir=None):
    """Aggregate and curate reduced/non reduced datasets"""
    X = []
    keys = []
    for dataset in datasets:
        if source == 'unmasked':
            this_X = load(join(unmask_dir, dataset, 'imgs.pkl'))
        else:
            this_X = load(join(reduced_dir, source, dataset, 'Xt.pkl'))

        # Curation
        this_X = this_X.reset_index(level=['direction'], drop=True)
        if dataset == 'brainomics':
            this_X = this_X.drop(['effects_of_interest'], level='contrast')
        if dataset == 'brainpedia':
            contrasts = this_X.index.get_level_values('contrast').values
            indices = []
            for i, contrast in enumerate(contrasts):
                if contrast.endswith('baseline'):
                    indices.append(i)
            this_X = this_X.iloc[indices]
            for i, (sub_dataset, this_sub_X) in \
                    enumerate(this_X.groupby(level='dataset')):
                if sub_dataset == 'ds102':
                    continue
                this_sub_X = this_sub_X.loc[sub_dataset]
                X.append(this_sub_X.astype(np.float32))
                keys.append(sub_dataset)
        else:
            X.append(this_X)
            keys.append(dataset)
    X = pd.concat(X, keys=keys, names=['dataset'])
    X.sort_index(inplace=True)
    return X


def split_folds(X, test_size=0.2, train_size=None, random_state=None):
    X_train = []
    X_test = []
    datasets = X.index.get_level_values('dataset').unique().values
    if not isinstance(test_size, dict):
        test_size = {dataset: test_size for dataset in datasets}
    if not isinstance(train_size, dict):
        train_size = {dataset: train_size for dataset in datasets}

    for dataset, this_X in X.groupby(level='dataset'):
        subjects = this_X.index.get_level_values('subject').values
        if dataset in test_size:
            this_test_size = test_size[dataset]
        else:
            this_test_size = .5
        if dataset in train_size:
            this_train_size = train_size[dataset]
        else:
            this_train_size = .5
        cv = GroupShuffleSplit(n_splits=1,
                               test_size=this_test_size,
                               train_size=this_train_size,
                               random_state=random_state)
        train, test = next(cv.split(this_X, groups=subjects))
        X_train.append(this_X.iloc[train])
        X_test.append(this_X.iloc[test])
    # WTF autocast in pandas
    X_train = pd.concat(X_train, axis=0).astype(np.float32)
    X_test = pd.concat(X_test, axis=0).astype(np.float32)
    X_train.sort_index(inplace=True)
    X_test.sort_index(inplace=True)
    return X_train, X_test


class MultiDatasetTransformer(TransformerMixin):
    """Utility transformer"""
    def __init__(self, with_std=False, with_mean=True,
                 per_dataset=True, integer_coding=False):
        self.with_std = with_std
        self.with_mean = with_mean
        self.per_dataset = per_dataset
        self.integer_coding = integer_coding

    def fit(self, df):
        self.lbins_ = {}
        if self.per_dataset:
            self.scs_ = {}
        else:
            self.sc_ = StandardScaler(with_std=self.with_std,
                                with_mean=self.with_mean)
            self.sc_.fit(df.values)
        for dataset, sub_df in df.groupby(level='dataset'):
            if self.integer_coding:
                lbin = LabelEncoder()
            else:
                lbin = LabelBinarizer()
            this_y = sub_df.index.get_level_values('contrast')
            if self.per_dataset:
                sc = StandardScaler(with_std=self.with_std,
                                    with_mean=self.with_mean)
                sc.fit(sub_df.values)
                self.scs_[dataset] = sc
            lbin.fit(this_y)
            self.lbins_[dataset] = lbin
        return self

    def transform(self, df):
        X = []
        y = []
        if not self.per_dataset:
            df = df.copy()
            df[:] = self.sc_.transform(df.values)
        for dataset, sub_df in df.groupby(level='dataset'):
            lbin = self.lbins_[dataset]
            if self.per_dataset:
                sc = self.scs_[dataset]
                this_X = sc.transform(sub_df.values)
            else:
                this_X = sub_df.values
            this_y = sub_df.index.get_level_values('contrast')
            this_y = lbin.transform(this_y)
            if not self.integer_coding and this_y.shape[1] == 1:
                this_y = np.hstack([this_y, np.logical_not(this_y)])
            y.append(this_y)
            X.append(this_X)
        return tuple(X), tuple(y)

    def inverse_transform(self, df, ys):
        contrasts = []
        for (dataset, sub_df), this_y in zip(df.groupby(level='dataset'), ys):
            lbin = self.lbins_[dataset]
            these_contrasts = lbin.inverse_transform(this_y)
            these_contrasts = pd.Series(these_contrasts, index=sub_df.index)
            contrasts.append(these_contrasts)
        contrasts = pd.concat(contrasts, axis=0)
        return contrasts


def make_projection_matrix(bases, scale_bases=True):
    if not isinstance(bases, list):
        bases = [bases]
    proj = []
    rec = []
    for i, basis in enumerate(bases):
        if scale_bases:
            S = np.std(basis, axis=1)
            S[S == 0] = 1
            basis = basis / S[:, np.newaxis]
            proj.append(pinv(basis))
            rec.append(basis)
    proj = np.concatenate(proj, axis=1)
    rec = np.concatenate(rec, axis=0)
    proj_inv = np.linalg.inv(proj.T.dot(rec.T)).T.dot(rec)
    return proj, proj_inv, rec
