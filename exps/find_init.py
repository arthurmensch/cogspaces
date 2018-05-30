import numpy as np
from joblib import Memory
from nilearn.input_data import NiftiMasker
from numpy.linalg import lstsq
from os.path import expanduser

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask

modl_atlas = fetch_atlas_modl()
mask = fetch_mask()['hcp']
dict_512 = modl_atlas['components512']
dict_128 = modl_atlas['components128']
dict_64 = modl_atlas['components64']

mem = Memory(cachedir=expanduser('~/cache'))
masker = NiftiMasker(mask_img=mask, memory=mem).fit()
dict_512 = masker.transform(dict_512)
dict_128 = masker.transform(dict_128)
dict_64 = masker.transform(dict_64)

loadings = lstsq(dict_512.T, dict_64.T)
np.save('loadings_64', loadings)

loadings = np.load('loadings_64.npy')
z = loadings[0].T
print(z.shape)

print(np.sum(z, axis=1))