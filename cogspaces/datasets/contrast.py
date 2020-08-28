from os.path import join
from typing import List

import pandas as pd
from joblib import load
from nilearn.datasets import fetch_neurovault_ids

from cogspaces.datasets.derivative import STUDY_LIST, add_study_contrast

nv_ids = {'archi': 4339, 'hcp': 4337, 'brainomics': 4341, 'camcan': 4342,
          'la5c': 4343, 'brainpedia': 1952}


def _assemble(images, images_meta, study):
    records = []
    for image, meta in zip(images, images_meta):
        if study == 'brainpedia':
            this_study = meta['study']
            subject = meta['name'].split('_')[-1]
            contrast = '_'.join(meta['task'].split('_')[1:])
            task = meta['task'].split('_')[0]
        else:
            this_study = study
            subject = meta['name'].split('_')[0]
            contrast = meta['contrast_definition']
            task = meta['task']
        records.append([image, this_study, subject, task, contrast])
    df = pd.DataFrame(records, columns=['z_map', 'study', 'subject', 'task', 'contrast'])
    return df


def fetch_contrasts(studies: str or List[str] = 'all', data_dir=None):
    dfs = []
    if studies == 'all':
        studies = nv_ids.keys()
    for study in studies:
        if study not in nv_ids:
            return ValueError('Wrong dataset.')
        data = fetch_neurovault_ids([nv_ids[study]], data_dir=data_dir, verbose=10,
                                    mode='download_new')
        dfs.append(_assemble(data['images'], data['images_meta'], study))
    return pd.concat(dfs)


def load_masked_contrasts(data_dir):
    Xs, ys = {}, {}
    for study in STUDY_LIST:
        Xs[study], ys[study] = load(join(data_dir, 'masked',
                                         'data_%s.npy'), mmap_mode='r')
    ys = add_study_contrast(ys)
    return Xs, ys


