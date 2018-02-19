import os
import re
from os.path import join

import pandas as pd
import torch
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

idx = pd.IndexSlice


class MultiStandardScaler(BaseEstimator, TransformerMixin):
    """Simple wrapper around StandardScaler to handle multipe datasets.

    Attributes
    ----------
    self.sc_: dict, Dictionaries indexed by study, owning all StandardScaler
            for each study

    """
    def fit(self, data):
        self.sc_ = {}
        for study, this_data in data.items():
            self.sc_[study] = StandardScaler().fit(this_data)
        return self

    def transform(self, data):
        transformed = {}
        for study, this_data in data.items():
            transformed[study] = self.sc_[study].transform(this_data)
        return transformed

    def inverse_transform(self, data):
        transformed = {}
        for study, this_data in data.items():
            transformed[study] = self.sc_[study].inverse_transform(this_data)
        return transformed


class MultiTargetEncoder(BaseEstimator, TransformerMixin):
    def fit(self, targets):
        self.le_ = {}
        for study, target in targets.items():
            d = defaultdict(LabelEncoder)
            target.apply(lambda x: d[x.name].fit(x))
            self.le_[study] = d
        return self

    def transform(self, targets):
        res = {}
        for study, target in targets.items():
            d = self.le_[study]
            res[study] = target.apply(lambda x: d[x.name].transform(x))
        return res

    def inverse_transform(self, targets):
        res = {}
        for study, target in targets.items():
            d = self.le_[study]
            res[study] = target.apply(lambda x: d[x.name].inverse_transform(x))
        return res


class ImgContrastDataset(Dataset):
    def __init__(self, data, contrasts=None):
        if contrasts is not None:
            assert data.shape[0] == contrasts.shape[0]
        self.data = data
        self.contrasts = contrasts

    def __getitem__(self, index):
        data = self.data[index]
        data = torch.from_numpy(data).float()
        if self.contrasts is None:
            if data.ndimension() == 2:
                contrasts = torch.LongTensor((data.shape[0], 1)).fill_(-1)
            else:
                contrasts = - torch.LongTensor([-1])
        else:
            contrasts = self.contrasts.iloc[index]
            if hasattr(contrasts, 'values'):
                contrasts = torch.from_numpy(contrasts.values).long()
            else:
                contrasts = torch.LongTensor([int(contrasts)])
        return data, contrasts

    def __len__(self):
        return self.data.shape[0]


class RepeatedDataLoader(DataLoader):
    def __iter__(self):
        while True:
            for data in super().__iter__():
                yield data


def load_data(data_dir):
    expr = re.compile("data_(.*).pt")

    data = {}
    for file in os.listdir(data_dir):
        match = re.match(expr, file)
        if match:
            study = match.group(1)
            data[study] = load(join(data_dir, file), mmap_mode='r')
    return unzip_data(data)


def zip_data(data, target):
    return {study: (data[study], target[study]) for study in data}


def unzip_data(data):
    return {study: data[study][0] for study in data}, \
           {study: data[study][1] for study in data}


def train_test_split(data, target, test_size=.5,
                     random_state=0):
    data = zip_data(data, target)
    datasets = {'train': {}, 'test': {}}
    for study, (this_data, this_target) in data.items():
        cv = GroupShuffleSplit(n_splits=1, test_size=test_size,
                               random_state=random_state)
        train, test = next(cv.split(X=this_data,
                                    groups=this_target['subject']))
        for fold, indices in [('train', train), ('test', test)]:
            datasets[fold][study] = this_data[indices], \
                                    this_target.iloc[indices]
    train_data, train_target = unzip_data(datasets['train'])
    test_data, test_target = unzip_data(datasets['test'])
    return train_data, test_data, train_target, test_target
