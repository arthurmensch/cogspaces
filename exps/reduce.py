import os
from os.path import join
from typing import List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from nilearn.input_data import NiftiMasker
from sklearn.utils import gen_batches

from cogspaces.datasets import fetch_mask, fetch_atlas_modl, \
    fetch_contrasts, STUDY_LIST
from cogspaces.datasets.utils import get_data_dir
from cogspaces.raw_datasets.contrast import fetch_all

idx = pd.IndexSlice


def single_mask(masker, imgs):
    return masker.transform(imgs)


def single_reduce(components, data, lstsq=False):
    if not lstsq:
        return data.dot(components.T)
    else:
        X, _, _, _ = np.linalg.lstsq(components.T, data.T)
        return X.T


def mask_contrasts(studies: Union[str, List[str]] = 'all',
                   output_dir: str = 'masked',
                   use_raw=False,
                   n_jobs: int = 1):
    batch_size = 10

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if use_raw and studies == 'all':
        data = fetch_all()
    else:
        data = fetch_contrasts(studies)
    mask = fetch_mask()
    masker = NiftiMasker(smoothing_fwhm=4, mask_img=mask,
                         verbose=0, memory_level=1, memory=None).fit()

    for study, this_data in data.groupby('study'):
        imgs = this_data['z_map'].values
        targets = this_data.reset_index()

        n_samples = this_data.shape[0]
        batches = list(gen_batches(n_samples, batch_size))
        this_data = Parallel(n_jobs=n_jobs, verbose=10,
                             backend='multiprocessing', mmap_mode='r')(
            delayed(single_mask)(masker, imgs[batch]) for batch in batches)
        this_data = np.concatenate(this_data, axis=0)

        dump((this_data, targets), join(output_dir, 'data_%s.pt' % study))


def reduce_contrasts(components: str = 'components_453_gm',
                     studies: Union[str, List[str]] = 'all',
                     masked_dir='unmasked', output_dir='reduced',
                     n_jobs=1, lstsq=False, ):
    batch_size = 200

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if studies == 'all':
        studies = STUDY_LIST

    modl_atlas = fetch_atlas_modl()
    mask = fetch_mask()
    dictionary = modl_atlas[components]
    masker = NiftiMasker(mask_img=mask).fit()
    components = masker.transform(dictionary)
    for study in studies:
        this_data, targets = load(join(masked_dir, 'data_%s.pt' % study))
        n_samples = this_data.shape[0]
        batches = list(gen_batches(n_samples, batch_size))
        this_data = Parallel(n_jobs=n_jobs, verbose=10,
                             backend='multiprocessing', mmap_mode='r')(
            delayed(single_reduce)(components,
                                   this_data[batch], lstsq=lstsq)
            for batch in batches)
        this_data = np.concatenate(this_data, axis=0)

        dump((this_data, targets), join(output_dir,
                                        'data_%s.pt' % study))

n_jobs = 65

mask_contrasts(studies='all', use_raw=True, output_dir=join(get_data_dir(), 'loadings'), n_jobs=n_jobs)

reduce_contrasts(studies='all',
                 masked_dir=join(get_data_dir(), 'masked'),
                 output_dir=join(get_data_dir(), 'loadings'),
                 components='components_453_gm', n_jobs=n_jobs, lstsq=False)
