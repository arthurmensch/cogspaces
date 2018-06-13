import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from nilearn._utils import check_niimg
from nilearn.datasets import fetch_icbm152_2009
from nilearn.image import new_img_like
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi, plot_prob_atlas, plot_stat_map
from os.path import expanduser, join
from scipy.ndimage import grey_erosion

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask

modl_atlas = fetch_atlas_modl()
mask = fetch_mask()['hcp']
dict_512 = modl_atlas['components512']
# dict_128 = modl_atlas['components128_small']

mem = Memory(cachedir=expanduser('~/cache'))
masker = NiftiMasker(mask_img=mask, memory=mem).fit()
dict_512 = masker.transform(dict_512)
# dict_128 = masker.transform(dict_128)

wm = fetch_icbm152_2009()['wm']
csf = fetch_icbm152_2009()['csf']
wm = check_niimg(wm)
csf = check_niimg(csf)

wmcsf = wm.get_data() + csf.get_data()
wmcsf = grey_erosion(wmcsf, size=(2, 2, 2))
wmcsf = new_img_like(wm, data=wmcsf)

plot_stat_map(wmcsf)
plt.show()
wmcsf = masker.transform(wmcsf)
overlap = dict_512.dot(wmcsf[0])
keep = overlap < 0.5
remove = np.logical_not(keep)
curated_dict = dict_512[keep]
removed = dict_512[remove]
coverage = np.any(curated_dict, axis=0)
plot_roi(masker.inverse_transform(coverage))
plot_prob_atlas(masker.inverse_transform(removed),
                view_type='continuous')
print('Removed components: %i' % np.sum(remove))
plt.show()

curated_dict = masker.inverse_transform(curated_dict)
curated_dict.to_filename(join(modl_atlas['data_dir'],
                              'components_512_gm.nii.gz'))
