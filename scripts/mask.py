import os
from os.path import join

import numpy as np
from joblib import delayed, Parallel
from nilearn.input_data import NiftiMasker
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import gen_batches

from cogspaces.datasets.contrasts import fetch_all, fetch_archi
from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_data_dir


class Projection(BaseEstimator, TransformerMixin):
    def __init__(self, components='components512'):
        self.components = components

    def fit(self):
        modl_atlas = fetch_atlas_modl()
        mask = fetch_mask()
        dictionary = modl_atlas[self.components]
        masker = NiftiMasker(mask_img=mask).fit()
        self.components_ = masker.transform(dictionary)
        return self

    def transform(self, imgs):
        if isinstance(imgs, str):
            imgs = [imgs]
        X = []
        for img in imgs:
            with open(img, 'rb') as f:
                X.append(np.load(f))
        X = np.vstack(X)
        return X.dot(self.components_.T)


def unmask_single(transformer, imgs, data_dir, dest_dir,
                  create_structure=False,):
    if not create_structure:
        data = transformer.transform(imgs)
    for i, img in enumerate(imgs):
        dest = img.replace(data_dir, join(data_dir, dest_dir))
        if create_structure:
            dirname = os.path.dirname(dest)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        else:
            with open(dest, 'wb+') as f:
                np.save(f, data[i])


def mask(transformer, data_dir=None, dest_dir='masked', batch_size=1,
         n_jobs=30):
    data_dir = get_data_dir(data_dir)
    contrasts = fetch_archi(data_dir)
    imgs = contrasts['z_map'].values
    n_samples = imgs.shape[0]
    batches = list(gen_batches(n_samples, batch_size))
    unmask_single(transformer, imgs, data_dir, dest_dir,
                  create_structure=True)
    Parallel(n_jobs=n_jobs, verbose=10)(delayed(unmask_single)(
        transformer, imgs[batch], data_dir,
        dest_dir) for batch in batches)


# mask = fetch_mask()
# transformer = NiftiMasker(smoothing_fwhm=4, mask_img=mask,
#                      verbose=0, memory_level=1, memory=None).fit()
# mask(transformer, dest_dir='masked', project=None)
transformer = Projection('components512').fit()
mask(transformer, data_dir=join(get_data_dir(), 'masked'),
     batch_size=256, dest_dir='masked_512',)
