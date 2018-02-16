import os
from os.path import join

import numpy as np
from joblib import delayed, Parallel
from nilearn.input_data import NiftiMasker
from sklearn.utils import gen_batches

from cogspaces.datasets.contrasts import fetch_all
from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_data_dir


def get_project(components='components512'):
    modl_atlas = fetch_atlas_modl()
    mask = fetch_mask()
    dictionary = modl_atlas[components]
    masker = NiftiMasker(mask_img=mask).fit()
    weights = masker.transform(dictionary)
    return weights


def unmask_single(masker, imgs, data_dir, dest_dir, project=None,
                  create_structure=False,):
    if not create_structure:
        data = masker.transform(imgs)
        if project is not None:
            data = data.dot(project.T)
    for i, img in enumerate(imgs):
        dest = img.replace(data_dir, join(data_dir, dest_dir))
        if create_structure:
            dirname = os.path.dirname(dest)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        else:
            with open(dest, 'wb+') as f:
                np.save(f, data[i])


def unmask(data_dir=None, dest_dir='masked', project=None, n_jobs=30):
    data_dir = get_data_dir(data_dir)
    contrasts = fetch_all()
    mask = fetch_mask()
    masker = NiftiMasker(smoothing_fwhm=4, mask_img=mask,
                         verbose=0, memory_level=1, memory=None).fit()
    imgs = contrasts['z_map'].values
    n_samples = imgs.shape[0]
    batches = list(gen_batches(n_samples, 32))
    unmask_single(masker, imgs, data_dir, dest_dir, project=project,
                  create_structure=True)
    Parallel(n_jobs=n_jobs, verbose=10)(delayed(unmask_single)(
        masker, imgs[batch], data_dir,
        dest_dir, project=project) for batch in batches)


unmask(dest_dir='masked_512', project=get_project('components512'))
