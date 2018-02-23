import os
import re
from os.path import join

import pandas as pd
import torch
from joblib import load
from torch.utils.data import Dataset, DataLoader

from cogspaces.utils import unzip_data

idx = pd.IndexSlice


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


def load_data_from_dir(data_dir):
    expr = re.compile("data_(.*).pt")

    data = {}
    for file in os.listdir(data_dir):
        match = re.match(expr, file)
        if match:
            study = match.group(1)
            data[study] = load(join(data_dir, file), mmap_mode='r')
    return unzip_data(data)


