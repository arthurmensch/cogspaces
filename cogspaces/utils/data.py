import os
import re
from os.path import join

import torch
from joblib import load
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    def __init__(self, data_mmap, target_mmap):
        assert data_mmap.shape[0] == target_mmap.shape[0]
        self.data_mmap = data_mmap
        self.target_mmap = target_mmap

    def __getitem__(self, index):
        return torch.from_numpy(self.data_mmap[index]), \
               torch.from_numpy(self.target_mmap[index])

    def __len__(self):
        return self.data_mmap.shape[0]

    def target_size(self):
        return self.target_mmap.max() + 1

    def in_features(self):
        return self.data_mmap.shape[0]


def load_dataset(prepare_dir):
    expr = re.compile("data_(.*)_(.*)\.pt")

    datasets = {'train': {}, 'test': {}}
    for file in os.listdir(prepare_dir):
        match = re.match(expr, file)
        if match:
            dataset = match.group(1)
            fold = match.group(2)
            X, y, lenc = load(join(prepare_dir, file), mmap_mode='r')
            datasets[fold][dataset] = MemmapDataset(X, y)
    return datasets