import matplotlib.pyplot as plt
import numpy as np
from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask
from joblib import Memory
from nilearn._utils import check_niimg
from nilearn.image import concat_imgs
from nilearn.image import swap_img_hemispheres, iter_img
from nilearn.input_data import NiftiMasker
from os.path import expanduser

modl_atlas = fetch_atlas_modl()
mask = fetch_mask()['hcp']
dict_512 = modl_atlas['components512']

mem = Memory(cachedir=expanduser('~/cache'))
masker = NiftiMasker(mask_img=mask, memory=mem).fit()

dict_512 = check_niimg(dict_512)
dict_512.get_data()
dict_512_sym = concat_imgs(list(map(swap_img_hemispheres, iter_img(dict_512))))
dict_512_sym = masker.transform(dict_512_sym)

dict_512 = masker.transform(dict_512)

corr = dict_512.dot(dict_512_sym.T)

assign = np.argmax(corr, axis=1)

dict_512_sym = dict_512_sym[assign]

corr = dict_512.dot(dict_512_sym.T)

np.save('assign', assign)

plt.figure()
plt.imshow(corr)
plt.show()
