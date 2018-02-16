import re
import os
from os.path import join
from joblib import load

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

idx = pd.IndexSlice


class tfMRIDataset(Dataset):
    def __init__(self, data, contrasts,
                 subjects):
        assert data.shape[0] == contrasts.shape[0]
        assert subjects.shape[0] == contrasts.shape[0]
        self.data = data
        self.contrasts = contrasts
        self.subjects = subjects
        self.le_contrasts = LabelEncoder().fit(contrasts)
        self.le_subjects = LabelEncoder().fit(contrasts)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data[index])
        contrasts = self.le_contrasts.transform(self.contrasts[index])
        subjects = self.le_subjects.transform(self.subjects[index])
        contrasts = torch.LongTensor(contrasts)
        subjects = torch.LongTensor(subjects)
        return data, contrasts, subjects

    def __len__(self):
        return self.data.shape[0]

    def n_contrasts(self):
        return len(self.le_contrasts.classes_)

    def n_features(self):
        return self.data.shape[0]

    def copy_label_encoders(self, dataset):
        self.le_contrasts = dataset.le_contrasts
        self.le_subjects = dataset.le_subjects


def load_dataset(prepare_dir):
    expr = re.compile("data_(.*)_(.*)\.pt")

    datasets = {'train': {}, 'test': {}}
    for file in os.listdir(prepare_dir):
        match = re.match(expr, file)
        if match:
            dataset = match.group(1)
            fold = match.group(2)
            data = load(join(prepare_dir, file), mmap_mode='r')
            datasets[fold][dataset] = tfMRIDataset(*data)

    for dataset in datasets['test']:
        datasets['test'][dataset].copy_label_encoders(datasets['train']
                                                      [dataset])
    return datasets