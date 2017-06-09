import os
from os.path import join

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import load
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelBinarizer

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


def make_data_frame(datasets, source, n_subjects=None,
                    reduced_dir=None, unmask_dir=None):
    """Aggregate and curate reduced/non reduced datasets"""
    X = []
    if not isinstance(n_subjects, dict):
        n_subjects = {dataset: n_subjects for dataset in datasets}

    for dataset in datasets:
        if source == 'unmasked':
            this_X = load(join(unmask_dir, dataset, 'imgs.pkl'))
        else:
            this_X = load(join(reduced_dir, source, dataset, 'Xt.pkl'))

        # Curation
        if dataset in ['brainomics']:
            this_X = this_X.drop(['effects_of_interest'], level='contrast')
        this_X = this_X.reset_index(level=['direction'], drop=True)
        this_n_subjects = n_subjects[dataset]
        subjects = this_X.index.get_level_values('subject').unique().values
        subjects = subjects[:this_n_subjects]
        this_X = this_X.loc[idx[subjects.tolist()]]

        this_X -= this_X.mean(axis=0)
        # this_X /= this_X.std(axis=0)

        X.append(this_X)
    X = pd.concat(X, keys=datasets, names=['dataset'])
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
        cv = GroupShuffleSplit(n_splits=1,
                               test_size=test_size[dataset],
                               train_size=train_size[dataset],
                               random_state=random_state)
        train, test = next(cv.split(this_X, groups=subjects))
        X_train.append(this_X.iloc[train])
        X_test.append(this_X.iloc[test])
    X_train = pd.concat(X_train, axis=0)
    X_test = pd.concat(X_test, axis=0)
    X_train.sort_index(inplace=True)
    X_test.sort_index(inplace=True)
    return X_train, X_test


class MultiDatasetTransformer(TransformerMixin):
    """Utility transformer"""
    def fit(self, df):
        self.lbins_ = []
        for dataset, sub_df in df.groupby(level='dataset'):
            lbin = LabelBinarizer()
            this_y = sub_df.index.get_level_values('contrast')
            lbin.fit(this_y)
            self.lbins_.append(lbin)
        return self

    def transform(self, df):
        X = []
        y = []
        for (dataset, sub_df), lbin in zip(df.groupby(level='dataset'),
                                           self.lbins_):
            this_X = sub_df.values
            this_y = sub_df.index.get_level_values('contrast')
            this_y = lbin.transform(this_y)
            y.append(this_y)
            X.append(this_X)
        return tuple(X), tuple(y)

    def inverse_transform(self, df, y):
        contrasts = []
        for (dataset, sub_df), this_y, lbin in zip(df.groupby(level='dataset'),
                                                   y, self.lbins_):
            these_contrasts = lbin.inverse_transform(this_y)
            these_contrasts = pd.Series(these_contrasts, index=sub_df.index)
            contrasts.append(these_contrasts)
        contrasts = pd.concat(contrasts, axis=0)
        return contrasts
