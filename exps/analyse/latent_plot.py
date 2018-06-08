import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_prob_atlas
from os.path import join, expanduser
from sklearn.utils.linear_assignment_ import linear_assignment

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask


def get_ref_comp(components):
    ref_components = fetch_atlas_modl()['components128']

    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()

    X_ref = masker.transform(ref_components)
    X = masker.transform(components)

    corr = (X.dot(X_ref.T) / np.sqrt(np.sum(X_ref ** 2, axis=1))[None, :]
            / np.sqrt(np.sum(X ** 2, axis=1))[None, :])
    assign = linear_assignment(-corr)[:, 1]
    X_ref = X_ref[assign]
    ref_components = masker.inverse_transform(X_ref)
    components = masker.inverse_transform(X)
    return ref_components, components

mem = Memory(cachedir=expanduser('~/cache'))

components = join('/home/arthur/output/cogspaces/dl_rest_37194.nii.gz')
ref_components, components = mem.cache(get_ref_comp)(components)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
plot_prob_atlas(components, figure=fig, axes=ax1)
plot_prob_atlas(ref_components, threshold='99.99%',
                figure=fig, axes=ax2)
plt.show()

# indices = [1, 2, 3]
#
# for index in indices:
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
#     component = index_img(components, index)
#     ref_component = index_img(ref_components, index)
#     vmax = np.abs(component.get_data()).max()
#     cut_coords = find_xyz_cut_coords(component, activation_threshold=vmax/ 3)
#     plot_glass_brain(component, figure=fig, axes=ax1, plot_abs=False)
#     plot_glass_brain(ref_component, figure=fig, axes=ax2, plot_abs=False)





# fsaverage = datasets.fetch_surf_fsaverage5()

# fig = plt.figure(figsize=(10, 5))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
#
# texture = surface.vol_to_surf(index_img(components, 25), fsaverage.pial_right)
# plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
#                    bg_map=fsaverage.sulc_right, threshold=1e-4,
#                    fig=fig, axes=ax1,
#                    cmap='cold_hot')
# texture = surface.vol_to_surf(index_img(components, 25), fsaverage.pial_left)
# plot_surf_stat_map(fsaverage.infl_left, texture, hemi='left',
#                    bg_map=fsaverage.sulc_right, threshold=1e-4,
#                    fig=fig, axes=ax2,
#                    cmap='cold_hot')
#
#
# from sklearn.decomposition import fastica
#
# K, W, S = fastica(X.T)
#
#
# from nilearn.plotting import plot_stat_map
#
# from nilearn.plotting import find_xyz_cut_coords
#
#
# i = 40
# this_img = index_img(img_independant, i)
# vmax = this_img.get_data().max()
#
# cut_coords = find_xyz_cut_coords(this_img, activation_threshold=vmax / 3)
# plot_stat_map(this_img, threshold=0, cut_coords=cut_coords)
#
# fig = plt.figure(figsize=(8, 3))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')
#
# texture = surface.vol_to_surf(this_img, fsaverage.pial_left)
# plot_surf_stat_map(fsaverage.infl_left, texture, hemi='left',
#                    bg_map=fsaverage.sulc_right, threshold=vmax / 6,
#                    fig=fig, axes=ax2,
#                    cmap='cold_hot')
# texture = surface.vol_to_surf(this_img, fsaverage.pial_right)
# plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
#                    bg_map=fsaverage.sulc_right, threshold=vmax / 6,
#                    fig=fig, axes=ax1,
#                    cmap='cold_hot')