import os
from os.path import join
from typing import List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from nilearn.input_data import NiftiMasker
from sklearn.utils import gen_batches

from cogspaces.datasets import fetch_mask, fetch_atlas_modl, \
    fetch_contrasts

idx = pd.IndexSlice


def single_mask(masker, imgs):
    return masker.transform(imgs)


def single_reduce(components, data, lstsq=False):
    if not lstsq:
        return data.dot(components.T)
    else:
        X, _, _, _ = np.linalg.lstsq(components.T, data.T)
        return X.T


def mask_contrasts(studies: Union[str, List[str]] ='all',
                   output_dir: str = 'masked',
                   n_jobs: int = 1):
    batch_size = 10

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = fetch_contrasts(studies)
    mask = fetch_mask()
    masker = NiftiMasker(smoothing_fwhm=4, mask_img=mask,
                         verbose=0, memory_level=1, memory=None).fit()

    for study, this_data in data.groupby('study'):
        imgs = this_data['z_map'].values
        targets = this_data.reset_index()[['study', 'subject', 'contrast']]

        n_samples = this_data.shape[0]
        batches = list(gen_batches(n_samples, batch_size))
        this_data = Parallel(n_jobs=n_jobs, verbose=10,
                             backend='multiprocessing', mmap_mode='r')(
            delayed(single_mask)(masker, imgs[batch]) for batch in batches)
        this_data = np.concatenate(this_data, axis=0)

        dump((this_data, targets), join(output_dir, 'data_%s.pt' % study))


def reduce_contrasts(components: str = 'components_512_gm',
                     studies: Union[str, List[str]] = 'all',
                     masked_dir='unmasked', output_dir='reduced',
                     n_jobs=1, lstsq=False, ):
    batch_size = 200

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    modl_atlas = fetch_atlas_modl()
    mask = fetch_mask()
    dictionary = modl_atlas[components]
    masker = NiftiMasker(mask_img=mask).fit()
    components = masker.transform(dictionary)
    for study in studies:
        this_data, targets = load(join(masked_dir, 'masked_%s.pt' % study))
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


mask_contrasts(studies=['brainpedia'], output_dir='masked')

reduce_contrasts(studies=['brainpedia'],
                 masked_dir='masked',
                 output_dir='reduced',
                 components='components_512_gm', n_jobs=2, lstsq=False)
