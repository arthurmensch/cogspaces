import os
from os.path import join

import numpy as np
import pandas as pd
from cogspaces.datasets.contrasts import fetch_all
from cogspaces.datasets.utils import get_data_dir, fetch_mask
from joblib import Parallel, delayed, dump
from nilearn.input_data import NiftiMasker
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches

idx = pd.IndexSlice

seed = 0


def mask(masker, imgs):
    return masker.transform(imgs)


data_dir = get_data_dir()
prepare_dir = join(data_dir, 'prepared_seed_%i' % seed)
if not os.path.exists(prepare_dir):
    os.makedirs(prepare_dir)
contrasts = fetch_all()
mask = fetch_mask()
masker = NiftiMasker(smoothing_fwhm=4, mask_img=mask,
                     verbose=0, memory_level=1, memory=None).fit()
n_jobs = 30
batch_size = 10
n_features = masker.mask_img_.get_data().sum()

new_contrasts = []
for dataset, these_contrasts in contrasts.groupby('dataset'):
    subjects = these_contrasts.index.get_level_values('subject').unique().values
    train_subjects, test_subjects = train_test_split(subjects, test_size=.5,
                                                     random_state=seed)
    train_contrasts = these_contrasts.loc[idx[:, train_subjects], :]
    test_contrasts = these_contrasts.loc[idx[:, test_subjects], :]
    these_contrasts = pd.concat([train_contrasts, test_contrasts], keys=
                                ['train', 'test'], names=['fold'])
    these_contrasts = these_contrasts.swaplevel('fold', 'dataset')
    new_contrasts.append(these_contrasts)
contrasts = pd.concat(new_contrasts)

for (dataset, fold), these_contrasts in contrasts.groupby(['dataset', 'fold']):
    imgs = these_contrasts['z_map'].values
    targets = these_contrasts.index.get_level_values('contrast').values
    subjects = these_contrasts.index.get_level_values('subject').values

    n_samples = these_contrasts.shape[0]
    batches = list(gen_batches(n_samples, batch_size))
    data = Parallel(n_jobs=n_jobs, verbose=10,
                    backend='multiprocessing', mmap_mode='r')(
        delayed(mask)(masker, imgs[batch]) for batch in batches)
    data = np.concatenate(data, axis=0)

    dump((data, targets, subjects),
         join(prepare_dir, 'data_%s_%s.pt' % (dataset, fold)))
