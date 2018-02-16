import numpy as np
import pandas as pd
import torch
from nilearn.input_data import NiftiMasker
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

idx = pd.IndexSlice


class tfMRITargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

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
    def __init__(self, data, mask=None):
        self.data = data.copy()
        self.data = self.data.sort_index(inplace=True)
        self.mask = mask
        if self.mask is not None:
            self.masker_ = NiftiMasker(smoothing_fwhm=4, mask_img=mask,
                                       verbose=0, memory_level=1,
                                       memory=None).fit()

    def __getitem__(self, index):
        if not hasattr(self, 'target_encoder_'):
            raise ValueError('Call self.set_target_encoder() first.')
        if isinstance(index, int):
            index = [index]
        img = self.data.iloc[index]
        filenames = img['z_map'].values
        if hasattr(self, 'masker'):
            data = self.masker.transform(filenames)
        else:
            data = []
            for filename in filenames:
                with open(filename, 'rb') as f:
                    this_data = np.load(f)
                    data.append(this_data)
            data = np.vstack(data)
        data = torch.from_numpy(data).float()

        targets = img.reset_index()[['study', 'subject', 'contrast']]
        studies, subjects, contrasts = self.target_encoder_.transform(targets)
        studies = torch.from_numpy(studies)
        subjects = torch.from_numpy(subjects)
        contrasts = torch.from_numpy(contrasts)

        return studies, subjects, contrasts, data

    def __len__(self):
        return len(self.data)

    def set_target_encoder(self, target_encoder=None):
        if target_encoder is None:
            targets = self.data.reset_index()[['study',
                                               'subject', 'contrast']]
            self.target_encoder_ = tfMRITargetEncoder().fit(targets)
        else:
            self.target_encoder_ = target_encoder

    def train_test_split(self, test_size=.5, random_state=None):
        def compute_fold(group):
            subjects = group.index.get_level_values('subject').unique().values
            train_subjects, test_subjects = train_test_split(
                subjects, test_size=test_size, random_state=random_state)
            fold = pd.Series(data='train', index=group.index)
            fold.loc[idx[:, test_subjects]] = 'test'
            group['fold'] = fold
            return group
        self.data = self.data.groupby('study').apply(compute_fold)

        res = {}
        for fold, subdata in self.data.groupby(by='fold'):
            dataset = tfMRIDataset(subdata.drop('fold', axis=1), self.mask)
            dataset.set_target_encoder(self.target_encoder_)
            res[fold] = dataset
        self.data.drop('fold', axis=1)
        return res['train'], res['test']

    def study_split(self):
        res = {}
        for name, sub_data in self.data.groupby('study'):
            dataset = tfMRIDataset(sub_data, self.mask)
            dataset.set_target_encoder(self.target_encoder_)
            res[name] = dataset
        return res