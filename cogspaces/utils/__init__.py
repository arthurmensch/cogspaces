from nilearn.input_data import NiftiMasker

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask


def zip_data(data, target):
    return {study: (data[study], target[study]) for study in data}


def unzip_data(data):
    return {study: data[study][0] for study in data}, \
           {study: data[study][1] for study in data}


def get_dictionary():
    modl_atlas = fetch_atlas_modl()
    dictionary = modl_atlas['components512_gm']
    masker = get_masker()
    dictionary = masker.transform(dictionary)
    return dictionary


def get_masker():
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    return masker