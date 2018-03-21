import os
import re
from os.path import join

import pandas as pd
import torch
from joblib import load
from torch.utils.data import Dataset, DataLoader

from cogspaces.utils import unzip_data

idx = pd.IndexSlice


class NiftiTargetDataset(Dataset):
    def __init__(self, data, targets=None):
        if targets is not None:
            assert data.shape[0] == targets.shape[0]
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        single = isinstance(index, int)
        data = self.data[index]
        data = torch.from_numpy(data).float()
        if self.targets is None:
            if single:
                study_targets = torch.LongTensor((1,)).fill_(0)
                targets = torch.LongTensor((1,)).fill_(0)
            else:
                study_targets = torch.LongTensor((data.shape[0], 1)).fill_(0)
                targets = torch.LongTensor((data.shape[0], 1)).fill_(0)
        else:
            targets = self.targets.iloc[index]['contrast']
            study_targets = self.targets.iloc[index]['study']
            if not single:
                targets = torch.from_numpy(targets.values).long()
                study_targets = torch.from_numpy(study_targets.values).long()
        return data, study_targets, targets

    def __len__(self):
        return self.data.shape[0]


def infinite_iter(iterable):
    while True:
        for elem in iterable:
            yield elem


def load_data_from_dir(data_dir):
    expr = re.compile("data_(.*).pt")

    data = {}
    for file in os.listdir(data_dir):
        match = re.match(expr, file)
        if match:
            study = match.group(1)
            data[study] = load(join(data_dir, file), mmap_mode='r')
    return unzip_data(data)
