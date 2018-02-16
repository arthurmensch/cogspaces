import os
import re
from os.path import join

import torch
from joblib import load
from nilearn.input_data import NiftiMasker
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class tfMRIIndexEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, data):
        datasets = data['dataset']
        self.dataset_le_ = LabelEncoder().fit(datasets)
        self.le_ = {}
        for dataset, sub_data in data.groupby(by='dataset'):
            contrasts = sub_data['contrast']
            subjects = sub_data['subject']
            self.le_[dataset] = {'contrast': LabelEncoder().fit(contrasts),
                                 'subject': LabelEncoder().fit(subjects)}
        return self

    def transform(self, data):
        def group_transform(group):
            dataset = group['dataset'][0]
            group['contrast'] = self.le_[dataset][
                'contrast'].transform(group['contrast'])
            group['subject'] = self.le_[dataset][
                'subject'].transform(group['subject'])
            return group

        data = data.groupby(by='dataset').apply(group_transform)
        contrasts = data['contrast'].values
        subjects = data['subject'].values
        datasets = data['dataset'].values
        datasets = self.dataset_le_.transform(datasets)
        return datasets, subjects, contrasts

    def inverse_transform(self, datasets, subjects, contrasts):
        def group_inverse_transform(group):
            dataset = group['dataset'][0]
            group['contrast'] = self.le_[dataset][
                'contrast'].inverse_transform(group['contrast'])
            group['subject'] = self.le_[dataset][
                'subject'].inverse_transform(group['subject'])
            return group

        datasets = self.dataset_le_.inverse_transform(datasets)
        data = pd.DataFrame(
            {'dataset': datasets, 'subject': subjects, 'contrast': contrasts})
        data = data.groupby(by='dataset').apply(group_inverse_transform)
        return data


class tfMRIDataset(Dataset):
    def __init__(self, data, mask):
        self.masker = NiftiMasker(smoothing_fwhm=4, mask_img=mask,
                                  verbose=0, memory_level=1,
                                  memory=None).fit()
        self.data = data.reset_index()
        self.ie = tfMRIIndexEncoder().fit(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = [index]
        img = self.data.iloc[index]
        filenames = img['z_map'].values
        datasets, subjects, contrasts = self.ie.transform(img)
        datasets = torch.from_numpy(datasets)
        subjects = torch.from_numpy(subjects)
        contrasts = torch.from_numpy(contrasts)
        data = torch.from_numpy(self.masker.transform(filenames)).float()
        return datasets, subjects, contrasts, data

    def __len__(self):
        return len(self.data)


class BatchtfMRIDataset(Dataset):
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
