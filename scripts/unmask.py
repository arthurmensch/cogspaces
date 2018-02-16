import os
from os.path import join

import numpy as np
from joblib import delayed, Parallel
from nilearn.input_data import NiftiMasker
from sklearn.utils import gen_batches

from cogspaces.datasets.contrasts import fetch_all, replace_filename_unmask
from cogspaces.datasets.utils import fetch_mask, get_data_dir


def unmask_single(masker, imgs, create_structure=False):
    if not create_structure:
        data = masker.transform(imgs)
    for img in imgs:
        dest = replace_filename_unmask(img)
        if create_structure:
            dest_dir = os.path.dirname(dest)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
        else:
            with open(dest, 'wb+') as f:
                np.save(f, data)


def unmask(data_dir=None, unmasked_dir='unmasked',
           n_jobs=30):
    data_dir = get_data_dir(data_dir)
    unmasked_dir = join(data_dir, unmasked_dir)
    if not os.path.exists(unmasked_dir):
        os.makedirs(unmasked_dir)
    contrasts = fetch_all()
    mask = fetch_mask()
    masker = NiftiMasker(smoothing_fwhm=4, mask_img=mask,
                         verbose=0, memory_level=1, memory=None).fit()
    imgs = contrasts['z_map'].values
    n_samples = imgs.shape[0]
    batches = list(gen_batches(n_samples, 1))
    unmask_single(masker, imgs, create_structure=True)
    Parallel(n_jobs=n_jobs, verbose=10)(delayed(unmask_single)(masker,
                                                               imgs[batch])
                                        for batch in batches)


unmask()
