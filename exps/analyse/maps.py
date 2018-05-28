import json
import numpy as np
import os
import re
from joblib import load, Memory, dump, Parallel, delayed
from nilearn._utils import check_niimg
from nilearn.image import iter_img
from nilearn.input_data import NiftiMasker
from os.path import join, expanduser
from sklearn.linear_model.base import LinearRegression

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir


class DenoisingLinearRegresion(LinearRegression):
    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept,
                         normalize=False, copy_X=True, n_jobs=1)

    def inverse_transform(self, Xr):
        # Xr.shape = (n_samples, n_targets)
        if self.fit_intercept:
            Xr -= self.intercept_[None, :]
        gram = self.coef_.dot(self.coef_.T)
        Xr = np.linalg.solve(gram, Xr.T).T
        return Xr.dot(self.coef_)

    def denoise(self, X):
        return self.inverse_transform(self.transform(X))


def analyse_unsupervised():
    modl_atlas = fetch_atlas_modl()
    dictionary_img = check_niimg(modl_atlas['components512'])
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    dictionary = masker.transform(dictionary_img)
    lr = DenoisingLinearRegresion(fit_intercept=False)
    lr.coef_ = dictionary
    return lr


def analyse_baseline(output_dir):
    mem = Memory(cachedir=expanduser('~/cache'))

    lr1 = mem.cache(analyse_unsupervised)()

    regex = re.compile(r'[0-9]+$')
    lr3s = {}
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
        except FileNotFoundError:
            continue
        study = config['data']['studies']
        estimator = load(join(this_exp_dir, 'estimator.pkl'))
        target_encoder = load(join(this_exp_dir, 'target_encoder.pkl'))
        names = target_encoder.le_[study]['contrast'].classes_
        lr3 = DenoisingLinearRegresion()
        coef = estimator.coef_[study]
        lr3.coef_ = coef.dot(lr1.coef_)
        lr3.coef_ -= np.mean(lr3.coef_, axis=0, keepdims=True)
        lr3.names_ = names
        lr3s[study] = lr3
    lr3s = {study: lr3s[study] for study in sorted(lr3s)}
    return lr1, lr3s


def analyse(output_dir):
    mem = Memory(cachedir=expanduser('~/cache'))

    lr1 = mem.cache(analyse_unsupervised)()

    estimator = load(join(output_dir, 'estimator.pkl'))
    target_encoder = load(join(output_dir, 'target_encoder.pkl'))

    embedder_coef = estimator.module_.embedder.linear.sparse_weight.data.numpy()
    lr2 = DenoisingLinearRegresion()
    snr = np.exp(
        - .5 * estimator.module_.embedder.linear.log_alpha.data.numpy())
    snr *= np.sign(embedder_coef)
    lr2.snr_ = snr.dot(lr1.coef_)
    lr2.coef_ = embedder_coef.dot(lr1.coef_)
    lr2.intercept_ = estimator.module_.embedder.linear.bias.data.numpy()

    lr3s = {}
    for study, classifier in estimator.module_.classifiers.items():
        var = classifier.batch_norm.running_var.data.numpy()
        std = np.sqrt(var)
        mean = classifier.batch_norm.running_mean.data.numpy()
        classifier_coef = classifier.linear.weight.data.numpy()

        classifier_intercept = classifier_coef.dot(lr2.intercept_ / std - mean)
        classifier_intercept += classifier.linear.bias.data.numpy()

        classifier_coef /= std

        classifier_coef = classifier_coef.dot(lr2.coef_)
        lr3 = lr3s[study] = DenoisingLinearRegresion()
        lr3.coef_ = classifier_coef
        lr3.coef_ -= np.mean(lr3.coef_, axis=0, keepdims=True)
        lr3.intercept_ = classifier_intercept
        names = target_encoder.le_[study]['contrast'].classes_
        lr3.names_ = names
    dump((lr1, lr2, lr3s), 'estimators.pkl')
    return lr1, lr2, lr3s


def plot_single(img, name, output_dir):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_stat_map, find_xyz_cut_coords

    vmax = img.get_data().max()
    cut_coords = find_xyz_cut_coords(img, activation_threshold=vmax / 3)
    fig = plt.figure(figsize=(8, 8))
    plot_stat_map(img, figure=fig, threshold=0, cut_coords=cut_coords)
    plt.savefig(join(output_dir, '%s.png' % name))
    plt.close(fig)


def plot_double(img, img2, name, output_dir):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
    vmax = img.get_data().max()
    cut_coords = find_xyz_cut_coords(img, activation_threshold=vmax / 3)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    plot_stat_map(img, title=name, figure=fig, axes=axes[0], threshold=0,
                  cut_coords=cut_coords)
    plot_stat_map(img2, figure=fig, axes=axes[1],
                  threshold=0,
                  cut_coords=cut_coords)
    plt.savefig(join(output_dir, '%s.png' % name))
    plt.close(fig)


def plot_all(imgs, output_dir, name, n_jobs=1):
    Parallel(n_jobs=n_jobs)(
        delayed(plot_single)(img,
                             ('%s_%i' % (name, i)), output_dir)
        for i, img in
        enumerate(iter_img(imgs)))


def plot_face_to_face(imgs, imgs_baseline, names, output_dir, n_jobs=1):
    Parallel(n_jobs=n_jobs)(
        delayed(plot_double)(img, img_baseline,
                             '%3i_%s' % (i, name), output_dir)
        for i, (img, img_baseline, name) in
        enumerate(zip(iter_img(imgs), iter_img(imgs_baseline), names)))


def make_level3_imgs(lr3s):
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    coef = np.concatenate([lr3.coef_ for study, lr3 in lr3s.items()], axis=0)
    names = ['%s_%s' % (study, name) for study, lr3 in lr3s.items() for name in
             lr3.names_]
    img = masker.inverse_transform(coef)
    return img, names


def make_level12_imgs(lr):
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    coef = lr.coef_
    mean = coef.mean(axis=1)
    coef[mean < 0] *= -1
    img = masker.inverse_transform(coef)
    snr = masker.inverse_transform(lr.snr_)
    return img, snr


def compute_corr(lr3s, name, output_dir):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    coef = np.concatenate([lr3.coef_ for study, lr3 in lr3s.items()], axis=0)
    names = ['%s_%s' % (study, name) for study, lr3 in lr3s.items() for name in
             lr3.names_]
    coef /= np.sqrt(np.sum(coef ** 2, axis=1, keepdims=True))
    corr = coef.dot(coef.T)

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(corr, interpolation=None, vmax=1, vmin=-1)
    ax.set_ylabel(names)
    plt.savefig(join(output_dir, '%s.png' % name), rotation=0)
    plt.close(fig)

    n_studies = len(lr3s)
    lengths = np.array([len(lr3.coef_) for study, lr3 in lr3s.items()])
    cum_lengths = np.concatenate([np.array([0]), np.cumsum(lengths)])
    mean_corr = np.array([[np.mean(np.abs(corr[startx:stopx, starty:stopy]))
                           for starty, stopy in
                           zip(cum_lengths, cum_lengths[1:])]
                          for startx, stopx in
                          zip(cum_lengths, cum_lengths[1:])
                          ])
    iu = np.triu_indices(n_studies)
    mean_corr[iu] = 0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(mean_corr, interpolation=None, vmax=1, vmin=0)
    ax.set_yticks(np.arange(len(lr3s)))
    ax.set_yticklabels([study for study in lr3s.keys()])
    plt.savefig(join(output_dir, '%s_study.png' % name))
    plt.close(fig)


def introspect(output_dir, baseline=False):
    introspect_dir = join(output_dir, 'maps')
    plot_dir = join(introspect_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if baseline:
        lr1, lr3s = analyse_baseline(output_dir)
        dump((lr1, lr3s), join(introspect_dir, 'transformers.pkl'))
        (lr1, lr3s) = load(join(introspect_dir, 'transformers.pkl'))
        baseline_imgs, _ = make_level3_imgs(lr3s)
        baseline_imgs.to_filename(join(introspect_dir, 'classif.nii.gz'))
    else:
        lr1, lr2, lr3s = analyse(output_dir)
        dump((lr1, lr2, lr3s), join(introspect_dir, 'transformers.pkl'))
        (lr1, lr2, lr3s) = load(join(introspect_dir, 'transformers.pkl'))
        imgs, names = make_level3_imgs(lr3s)
        imgs.to_filename(join(introspect_dir, 'classif.nii.gz'))
        dump(names, join(introspect_dir, 'names.pkl'))
        imgs2, snrs2 = make_level12_imgs(lr2)
        snrs2.to_filename(join(introspect_dir, 'snr.nii.gz'))
        imgs2.to_filename(join(introspect_dir, 'components.nii.gz'))


def plot(output_dir, baseline_output_dir, plot_components=True,
         plot_classif=True, n_jobs=1):
    introspect_dir = join(output_dir, 'maps')
    baseline_introspect_dir = join(baseline_output_dir, 'maps')
    plot_dir = join(introspect_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if plot_components:
        components = check_niimg(join(introspect_dir, 'components.nii.gz'))
        plot_all(components, plot_dir, 'components', n_jobs=n_jobs)
        components = check_niimg(join(introspect_dir, 'snr.nii.gz'))
        plot_all(components, plot_dir, 'snr')
    if plot_classif:
        names = load(join(introspect_dir, 'names.pkl'))
        imgs = join(baseline_introspect_dir, 'classif.nii.gz')
        baseline_imgs = join(baseline_introspect_dir, 'classif.nii.gz')
        plot_face_to_face(imgs, baseline_imgs, names, plot_dir, n_jobs=n_jobs)


def introspect_and_plot(output_dir, n_jobs=1):
    introspect(output_dir, baseline=False)

    baseline_output_dir = join(get_output_dir(), 'baseline_logistic_refit')
    plot(output_dir, baseline_output_dir, n_jobs=n_jobs, plot_classif=False)


if __name__ == '__main__':
    # baseline_output_dir = join(get_output_dir(), 'baseline_logistic_refit')
    # introspect(baseline_output_dir, baseline=True)
    # #
    # output_dir = join(get_output_dir(), 'multi_studies', '1969')
    output_dir = join(get_output_dir(), 'multi_studies', '1983')
    introspect_and_plot(output_dir, n_jobs=3)
