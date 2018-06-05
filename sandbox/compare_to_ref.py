import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map
from os.path import expanduser, join
from scipy.linalg import orthogonal_procrustes
from sklearn.utils.linear_assignment_ import linear_assignment

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask


def relative_stability(comp1, comp2):
    comp1 = comp1 / np.sqrt(np.sum(comp1 ** 2, axis=1, keepdims=True))
    comp2 = comp2 / np.sqrt(np.sum(comp2 ** 2, axis=1, keepdims=True))
    Q = comp1.dot(comp2.T)
    assign = linear_assignment(-np.abs(Q))[:, 1]
    Q = Q[:, assign]
    return np.abs(np.diag(Q)).tolist(), assign.tolist()


mem = Memory(cachedir=expanduser('~/cache'))
output_dir = expanduser('~/output/cogspaces/multi_studies/452/maps')

components = join(output_dir, 'components.nii.gz')

mask = fetch_mask()['hcp']

masker = MultiNiftiMasker(mask_img=mask, memory=mem, n_jobs=10, memory_level=1).fit()

ref = fetch_atlas_modl()['components128']
ref = masker.transform(ref)

components = masker.transform(components)

score, assign = relative_stability(ref, components)
print(np.array(score).mean())
print(assign)

components = components[assign]

ref_n = ref - np.mean(ref, axis=0, keepdims=True)
components_n = components - np.mean(ref, axis=0, keepdims=True)
ref_n = ref_n / np.sqrt(np.sum(ref ** 2, axis=1, keepdims=True))
components_n = components_n / np.sqrt(np.sum(components ** 2, axis=1, keepdims=True))
R, _ = orthogonal_procrustes(ref_n.T, components_n.T)

corr_n = components_n.dot(ref_n.T)


rotated_components = R.dot(components)

corr = rotated_components.dot(ref.T)

plt.figure()
plt.imshow(corr_n)
plt.colorbar()


plt.figure()
plt.imshow(corr)
plt.colorbar()

plt.figure()

plot_stat_map(masker.inverse_transform(rotated_components[1]))
plot_stat_map(masker.inverse_transform(components[1]))
plot_stat_map(masker.inverse_transform(ref[1]))

plt.show()
