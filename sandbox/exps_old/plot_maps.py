import matplotlib
import numpy as np

matplotlib.use('pdf')

from os.path import join

from joblib import load, dump
from nilearn.image import iter_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords

from cogspaces.datasets import fetch_atlas_modl, fetch_mask
from cogspaces.pipeline import get_output_dir, make_projection_matrix
import matplotlib.pyplot as plt

def compute_rec():
    mask_img = fetch_mask()
    masker = MultiNiftiMasker(mask_img=mask_img).fit()
    atlas = fetch_atlas_modl()
    components_imgs = [atlas.positive_new_components16,
                       atlas.positive_new_components64,
                       atlas.positive_new_components512]
    components = masker.transform(components_imgs)
    proj, proj_inv, rec = make_projection_matrix(components, scale_bases=True)
    dump(rec, join(get_output_dir(), 'benchmark', 'rec.pkl'))


def load_rec():
    return load(join(get_output_dir(), 'benchmark', 'rec.pkl'))


# compute_rec()

exp_dirs = join(get_output_dir(), 'single_exp', '17')
models = []
rec = load_rec()
mask_img = fetch_mask()
masker = MultiNiftiMasker(mask_img=mask_img).fit()

for exp_dir in [exp_dirs]:
    estimator = load(join(exp_dirs, 'estimator.pkl'))
    transformer = load(join(exp_dirs, 'transformer.pkl'))
    for coef, (dataset, sc), (_, lbin) in zip(estimator.coef_, transformer.scs_.items(),
                                      transformer.lbins_.items()):
        print(dataset)
        classes = lbin.classes_
        print(classes)
        coef /= sc.scale_
        coef_rec = coef.dot(rec)
        # coef_rec -= np.mean(coef_rec, axis=0)
        print(join(exp_dirs, 'maps_%s.nii.gz' % dataset))
        imgs = masker.inverse_transform(coef_rec)
        imgs.to_filename(join(exp_dirs, 'maps_%s.nii.gz' % dataset))
        fig, axes = plt.subplots(len(classes) // 4, 4, figsize=(24, len(classes) // 4 * 3))
        axes = axes.ravel()
        for ax, img, this_class in zip(axes, iter_img(imgs), classes):
            this_class = this_class.replace('_', ' ')
            this_class = this_class.replace('&', ' ')
            vmax = np.abs(img.get_data()).max()
            coords = find_xyz_cut_coords(img, activation_threshold=vmax / 3)
            plot_stat_map(img, axes=ax, figure=fig, title=this_class, cut_coords=coords)
        fig.savefig(join(exp_dirs, 'maps_%s.png' % dataset))