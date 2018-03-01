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
        data = self.data[index]
        data = torch.from_numpy(data).float()
        if self.targets is None:
            if data.ndimension() == 2:
                targets = torch.LongTensor((data.shape[0], 3)).fill_(0)
            else:
                targets = - torch.LongTensor((3,)).fill_(0)
        else:
            targets = self.targets.iloc[index][['study', 'subject',
                                                'contrast']]
            targets = torch.from_numpy(targets.values).long()
        return data, targets

    def __len__(self):
        return self.data.shape[0]


class RepeatedDataLoader(DataLoader):
    def __iter__(self):
        while True:
            for data in super().__iter__():
                yield data

def load_data_from_dir(data_dir):
    expr = re.compile("data_(.*).pt")

    data = {}
    for file in os.listdir(data_dir):
        match = re.match(expr, file)
        if match:
            study = match.group(1)
            data[study] = load(join(data_dir, file), mmap_mode='r')
    return unzip_data(data)
