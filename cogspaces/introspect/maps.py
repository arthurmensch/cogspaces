import warnings

import numpy as np

warnings.filterwarnings('ignore', module='h5py', category=FutureWarning)
from nilearn.input_data import NiftiMasker

from cogspaces.datasets.utils import fetch_mask


def maps_from_model(estimator, dictionary, target_encoder, standard_scaler,
                    lstsq=False):
    coef, names = coefs_from_model(estimator, target_encoder,
                                   standard_scaler)
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    # components.shape = (n_components, n_voxels)
    dictionary = masker.transform(dictionary)
    if lstsq:
        gram = dictionary.dot(dictionary.T)
        dictionary = np.linalg.inv(gram).dot(dictionary)
    coef = {dictionary.dot(this_coef) for study, this_coef in coef.items()}
    # coef.shape = (n_components, n_classes)

    coef = {study: masker.inverse_transform(coef)
            for study, coef in coef.items()}
    return coef, names


def coefs_from_model(estimator, target_encoder, standard_scaler):
    coef = estimator.coef_
    scale = standard_scaler.scale_
    classes = target_encoder.classes_

    names = {}
    for study in coef:
        this_scale = scale[study]
        this_coef = coef[study]
        these_names = classes[study]
        this_coef = this_coef / this_scale[:, None]
        coef[study] = this_coef
        names[study] = these_names

    return coef, names
