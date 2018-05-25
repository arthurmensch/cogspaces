from os.path import join, expanduser


output_dir = join(expanduser('/home/arthur/output/cogspaces/compare/'))

components = join(output_dir, 'components.nii.gz')

from nilearn.plotting import plot_surf_stat_map
from nilearn.image import index_img
import matplotlib.pyplot as plt

from nilearn import surface
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage5()

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

texture = surface.vol_to_surf(index_img(components, 25), fsaverage.pial_right)
plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                   bg_map=fsaverage.sulc_right, threshold=1e-4,
                   fig=fig, axes=ax1,
                   cmap='cold_hot')
texture = surface.vol_to_surf(index_img(components, 25), fsaverage.pial_left)
plot_surf_stat_map(fsaverage.infl_left, texture, hemi='left',
                   bg_map=fsaverage.sulc_right, threshold=1e-4,
                   fig=fig, axes=ax2,
                   cmap='cold_hot')

from cogspaces.datasets.utils import fetch_mask
from nilearn.input_data import NiftiMasker

mask = fetch_mask()['hcp']
masker = NiftiMasker(mask_img=mask).fit()
X = masker.transform(components)

from sklearn.decomposition import fastica

K, W, S = fastica(X.T)

img_independant = masker.inverse_transform(S.T)

from nilearn.plotting import plot_stat_map

from nilearn.plotting import find_xyz_cut_coords


i = 40
this_img = index_img(img_independant, i)
vmax = this_img.get_data().max()

cut_coords = find_xyz_cut_coords(this_img, activation_threshold=vmax / 3)
plot_stat_map(this_img, threshold=0, cut_coords=cut_coords)

fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

texture = surface.vol_to_surf(this_img, fsaverage.pial_left)
plot_surf_stat_map(fsaverage.infl_left, texture, hemi='left',
                   bg_map=fsaverage.sulc_right, threshold=vmax / 6,
                   fig=fig, axes=ax2,
                   cmap='cold_hot')
texture = surface.vol_to_surf(this_img, fsaverage.pial_right)
plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                   bg_map=fsaverage.sulc_right, threshold=vmax / 6,
                   fig=fig, axes=ax1,
                   cmap='cold_hot')