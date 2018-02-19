import os
import re
from os.path import join

import pandas as pd
import torch
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset

idx = pd.IndexSlice


class tfMRITargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def target_sizes(self):
        return {study: len(le['contrast'].classes_) for study, le in
                self.le_.items()}

    def fit(self, data):
        studies = data['study']
        self.study_le_ = LabelEncoder().fit(studies)
        self.le_ = {}
        for study, sub_data in data.groupby(by='study'):
            contrasts = sub_data['contrast']
            subjects = sub_data['subject']
            self.le_[study] = {'contrast': LabelEncoder().fit(contrasts),
                               'subject': LabelEncoder().fit(subjects)}
        return self

    def transform(self, data):
        if isinstance(data, pd.Series):
            study = data['study']
            contrast = self.le_[study]['contrast'].transform(
                [data['contrast']])
            subject = self.le_[study]['subject'].transform(
                [data['subject']])
            study = self.study_le_.transform([study])
            return study, subject, contrast
        else:
            def group_transform(group):
                study = group['study'].iloc[0]
                group['contrast'] = self.le_[study][
                    'contrast'].transform(group['contrast'])
                group['subject'] = self.le_[study][
                    'subject'].transform(group['subject'])
                return group

            data = data.groupby(by='study').apply(group_transform)
            contrasts = data['contrast'].values
            subjects = data['subject'].values
            studies = data['study'].values
            studies = self.study_le_.transform(studies)
            return studies, subjects, contrasts

    def inverse_transform(self, studies, subjects, contrasts):
        def group_inverse_transform(group):
            study = group['study'][0]
            group['contrast'] = self.le_[study][
                'contrast'].inverse_transform(group['contrast'])
            group['subject'] = self.le_[study][
                'subject'].inverse_transform(group['subject'])
            return group

        studies = self.study_le_.inverse_transform(studies)
        data = pd.DataFrame(
            {'study': studies, 'subject': subjects, 'contrast': contrasts})
        data = data.groupby(by='study').apply(group_inverse_transform)
        return data


class tfMRIDataset(Dataset):
    def __init__(self, data, targets, target_encoder):
        assert data.shape[0] == targets.shape[0]
        self.data = data
        self.targets = targets
        self.target_encoder = target_encoder

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.target_encoder.transform(self.targets.iloc[index])
        data = torch.from_numpy(data).float()
        targets = tuple(torch.from_numpy(target).long() for target in targets)
        return data, targets

    def __len__(self):
        return self.data.shape[0]

    def n_features(self):
        return self[0][0].shape[0]


def load_prepared_data(data_dir, random_state=0, torch=True):
    expr = re.compile("data_(.*).pt")

    all_data = {}
    for file in os.listdir(data_dir):
        match = re.match(expr, file)
        if match:
            study = match.group(1)
            all_data[study] = load(join(data_dir, file), mmap_mode='r')

    all_targets = pd.concat(target for _, target in all_data.values())
    target_encoder = tfMRITargetEncoder().fit(all_targets)

    datasets = {}
    for study, (data, target) in all_data.items():
        cv = GroupShuffleSplit(n_splits=1, test_size=.5,
                               random_state=random_state)
        train, test = next(cv.split(X=data, groups=target['subject']))
        if torch:
            dataset = tfMRIDataset(data, target, target_encoder)
            datasets[study] = {fold: Subset(dataset, indices=indices)
                               for fold, indices
                               in [('train', train), ('test', test)]}
        else:
            datasets[study] = {fold: (data[indices], target.iloc[indices])
                               for fold, indices
                               in [('train', train), ('test', test)]}

    n_features = data.shape[1]
    target_sizes = target_encoder.target_sizes()
    return datasets, target_encoder, n_features, target_sizes
