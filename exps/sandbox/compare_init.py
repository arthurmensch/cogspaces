from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask
from joblib import Memory
from nilearn.input_data import NiftiMasker
from os.path import expanduser

modl_atlas = fetch_atlas_modl()
mask = fetch_mask()['hcp']
dict_512 = modl_atlas['components512']
dict_128 = modl_atlas['components128']


masker = NiftiMasker(mask_img=mask, memory=Memory(cachedir=expanduser('~/cache'))).fit()
# dict_512 = masker.transform(dict_512)
dict_128 = masker.transform(dict_128)
dict_finish = masker.transform(expanduser('~/plot_2009/maps/components.nii.gz'))

corr = dict_128.dot(dict_finish.T)
corr_self = dict_128.dot(dict_128.T)

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(corr)
plt.title('Corr')

plt.figure()
plt.imshow(corr_self)
plt.title('Self')

plt.show()