import matplotlib
matplotlib.use('Qt5Agg')

from os.path import join, expanduser

from joblib import load, Memory, dump
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map

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

exp_dirs = join(get_output_dir(), 'single_exp', '8')
models = []
rec = load_rec()
mask_img = fetch_mask()
masker = MultiNiftiMasker(mask_img=mask_img).fit()

for exp_dir in [exp_dirs]:
    estimator = load(join(exp_dirs, 'estimator.pkl'))
    transformer = load(join(exp_dirs, 'transformer.pkl'))
    print([(dataset, this_class) for dataset, lbin in transformer.lbins_.items()
         for this_class in lbin.classes_])
    coef = estimator.coef_
    coef_rec = coef.dot(rec)
    print(join(exp_dirs, 'maps.nii.gz'))
    imgs = masker.inverse_transform(coef_rec)
    imgs.to_filename(join(exp_dirs, 'maps.nii.gz'))
plot_stat_map(index_img(imgs, 10))
plt.show()