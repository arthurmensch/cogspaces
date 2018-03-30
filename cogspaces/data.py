import os
import re
from os.path import join

import pandas as pd
import torch
from joblib import load
from torch.utils.data import Dataset, DataLoader

from cogspaces.utils import unzip_data
import numpy as np

idx = pd.IndexSlice


class NiftiTargetDataset(Dataset):
    def __init__(self, data, targets=None):
        if targets is not None:
            assert data.shape[0] == targets.shape[0]
            self.study = targets['study'].values
            self.contrast = targets['contrast'].values
            self.all_contrast = targets['all_contrast'].values
        else:
            self.study = None
            self.contrast = None
            self.all_contrast = None

        self.data = data

    def __getitem__(self, index):
        single = isinstance(index, int)
        data = self.data[index]
        data = torch.from_numpy(data).float()
        if self.study is None:
            if single:
                study = torch.LongTensor((1,)).fill_(0)
                contrast = torch.LongTensor((1,)).fill_(0)
                all_contrast = torch.LongTensor((1,)).fill_(0)
            else:
                study = torch.LongTensor((data.shape[0], 1)).fill_(0)
                contrast = torch.LongTensor((data.shape[0], 1)).fill_(0)
                all_contrast = torch.LongTensor((1,)).fill_(0)
        else:
            contrast = self.contrast[index]
            all_contrast = self.all_contrast[index]
            study = self.study[index]
            if not single:
                contrast = torch.from_numpy(contrast.values)
                all_contrast = torch.from_numpy(all_contrast.values)
                study = torch.from_numpy(study.values)

        return data, study, contrast, all_contrast

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
            data[study] = np.asarray(load(join(data_dir, file)))
    return unzip_data(data)
