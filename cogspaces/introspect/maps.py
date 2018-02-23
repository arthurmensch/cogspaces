import warnings

import numpy as np

warnings.filterwarnings('ignore', module='h5py', category=FutureWarning)
from nilearn.input_data import NiftiMasker

from cogspaces.datasets.utils import fetch_mask


def maps_from_model(estimator,
                    dictionary,
                    target_encoder,
                    standard_scaler,
                    lstsq=False):
    mask = fetch_mask()
    masker = NiftiMasker(mask_img=mask).fit()
    # components.shape = (n_components, n_voxels)
    dictionary = masker.transform(dictionary)
    if lstsq:
        gram = dictionary.dot(dictionary.T)
        dictionary = np.linalg.inv(gram).dot(dictionary)
    # coef.shape = (n_components, n_classes)
    coef = estimator.coef_
    scale = standard_scaler.scale_
    classes = target_encoder.classes_

    transformed_coef = {}
    names = {}
    for study in coef:
        this_scale = scale[study]
        this_coef = coef[study]
        these_names = classes[study]
        this_coef = this_coef / this_scale[:, None]
        this_coef = this_coef.T.dot(dictionary)
        this_coef -= np.mean(this_coef, axis=0)[None, :]
        this_coef = masker.inverse_transform(this_coef)
        transformed_coef[study] = this_coef
        names[study] = these_names
    return transformed_coef, names