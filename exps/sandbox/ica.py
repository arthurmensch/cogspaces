from cogspaces.datasets.utils import fetch_mask
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from os.path import join, expanduser

output_dir = join(expanduser('~/output/cogspaces/multi_studies/1974/maps'))

components = join(output_dir, 'components.nii.gz')

mask = fetch_mask()['hcp']
masker = NiftiMasker(mask_img=mask).fit()
X = masker.transform(components)

from sklearn.decomposition import fastica

fsaverage = datasets.fetch_surf_fsaverage5()

K, W, S = fastica(X.T, whiten=False)

img_independant = masker.inverse_transform(S.T)
img_independant.to_filename(join(output_dir, 'independant.nii.gz'))
# i = 40
# this_img = index_img(img_independant, i)
# vmax = this_img.get_data().max()
#
# cut_coords = find_xyz_cut_coords(this_img, activation_threshold=vmax / 3)
#
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
#                    cmap='cold_hot');
# texture = surface.vol_to_surf(this_img, fsaverage.pial_right)
# plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
#                    bg_map=fsaverage.sulc_right, threshold=vmax / 6,
#                    fig=fig, axes=ax1,
#                    cmap='cold_hot');