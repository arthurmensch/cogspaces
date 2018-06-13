import numpy as np
from joblib import Memory
from nilearn.input_data import NiftiMasker
from numpy.linalg import lstsq
from os.path import expanduser, join

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir

modl_atlas = fetch_atlas_modl()
mask = fetch_mask()['hcp']
dict_512 = modl_atlas['components512_gm']
dict_128 = modl_atlas['components128']

mem = Memory(cachedir=expanduser('~/cache'))
masker = NiftiMasker(mask_img=mask, memory=mem).fit()
dict_512 = masker.transform(dict_512)
dict_128 = masker.transform(dict_128)

loadings, _, _, _ = lstsq(dict_512.T, dict_128.T)
loadings = loadings.T
output_dir = get_output_dir()
np.save(join(modl_atlas['data_dir'], 'loadings_128_gm.npy'), loadings)
