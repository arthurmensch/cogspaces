import numpy as np
from joblib import Memory
from nilearn.input_data import NiftiMasker
from os.path import expanduser

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask

modl_atlas = fetch_atlas_modl()
mask = fetch_mask()['hcp']
dict_512 = modl_atlas['components512']
dict_128 = modl_atlas['components128']


masker = NiftiMasker(mask_img=mask, memory=Memory(cachedir=expanduser('~/cache'))).fit()
dict_512 = masker.transform(dict_512)
dict_128 = masker.transform(dict_128)
#
# loadings = lstsq(dict_512.T, dict_128.T)
# np.save('loadings', loadings)

loadings = np.load('loadings.npy')
z = loadings[0].T
print(z.shape)

print(np.sum(z, axis=1))