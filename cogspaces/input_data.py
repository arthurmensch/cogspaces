from tempfile import NamedTemporaryFile

import numpy as np
# TODO: move
from joblib import Parallel, delayed
from nilearn.input_data import MultiNiftiMasker, NiftiMapsMasker
from nilearn.input_data.base_masker import filter_and_extract
from sklearn.utils import gen_batches


def make_array_data(imgs, mask=None, n_jobs=1, memory=None, maps=None):
    # if maps is None:
    n_samples = len(imgs)
    batch_size = 200
    batches = gen_batches(n_samples, batch_size)
    masker = MultiNiftiMasker(smoothing_fwhm=4, mask_img=mask,
                              n_jobs=n_jobs,
                              memory_level=1,
                              memory=memory).fit()
    if maps is not None:
        maps = masker.transform(maps)
        n_features = maps.shape[0]
    else:
        n_features = masker.mask_img_.get_data().sum()
    target = np.empty(shape=(n_samples, n_features), dtype=np.float32)
    for i, batch in enumerate(list(batches)[:2]):
        print('sample %i / %i' % (i * batch_size, n_samples))
        these_imgs = imgs[batch]
        masked = masker.transform(these_imgs)
        if maps is not None:
            masked = masked.dot(maps.T)
        target[batch] = masked
    return target




    # else:
    #     masker = NiftiMapsMasker(maps, smoothing_fwhm=4,
    #                              mask_img=mask,
    #                              memory_level=1,
    #                              memory=memory).fit()
    #     masked = Parallel(n_jobs=n_jobs)(
    #         delayed(masker.transform)(
    #             [this_data]) for img in imgs)
    #     X = np.concatenate(masked, axis=0)