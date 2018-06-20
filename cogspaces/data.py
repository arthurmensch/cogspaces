import itertools

import numpy as np
import os
import pandas as pd
import re
import torch
from joblib import load
from os.path import join
from sklearn.utils import check_random_state
from torch.utils.data import Dataset, DataLoader

from cogspaces.utils import unzip_data

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


def load_data_from_dir(data_dir):
    expr = re.compile("data_(.*).pt")

    data = {}
    for file in os.listdir(data_dir):
        match = re.match(expr, file)
        if match:
            study = match.group(1)
            # this_data, this_target = load(join(data_dir, file))
            # this_data = np.asarray(this_data)
            # data[study] = this_data, this_target
            data[study] = load(join(data_dir, file), mmap_mode='r')
    return unzip_data(data)


def infinite_iter(iterable):
    while True:
        for elem in iterable:
            yield elem


class RandomChoiceIter:
    def __init__(self, choices, p, seed=None):
        self.random_state = check_random_state(seed)
        self.choices = choices
        self.p = p

    def __next__(self):
        return self.random_state.choice(self.choices, p=self.p)


class MultiStudyLoaderIter:
    def __init__(self, loader):
        studies = loader.studies
        loaders = {study: DataLoader(data,
                                     shuffle=True,
                                     batch_size=loader.batch_size,
                                     pin_memory=loader.device.type == 'cuda')
                   for study, data in studies.items()}
        self.loader_iters = {study: infinite_iter(loader)
                             for study, loader in loaders.items()}

        studies = list(studies.keys())
        self.sampling = loader.sampling
        if self.sampling == 'random':
            p = np.array([loader.study_weights[study] for study in studies])
            assert (np.all(p >= 0))
            p /= np.sum(p)
            self.study_iter = RandomChoiceIter(studies, p, loader.seed)
        elif self.sampling == 'cycle':
            self.study_iter = itertools.cycle(studies)
        elif self.sampling == 'all':
            self.studies = studies
        else:
            raise ValueError('Wrong value for `sampling`')

        self.device = loader.device

    def __next__(self):
        inputs, targets = {}, {}
        if self.sampling == 'all':
            for study in self.studies:
                input, target = next(self.loader_iters[study])
                input = input.to(device=self.device)
                target = target.to(device=self.device)
                inputs[study], targets[study] = input, target
        else:
            study = next(self.study_iter)
            input, target = next(self.loader_iters[study])
            input = input.to(device=self.device)
            target = target.to(device=self.device)
            inputs[study], targets[study] = input, target
        return inputs, targets


class MultiStudyLoader:
    def __init__(self, studies,
                 batch_size=128, sampling='cycle',
                 study_weights=None, seed=None, device=-1):
        self.studies = studies
        self.batch_size = batch_size
        self.sampling = sampling
        self.study_weights = study_weights

        self.device = device
        self.seed = seed

    def __iter__(self):
        return MultiStudyLoaderIter(self)