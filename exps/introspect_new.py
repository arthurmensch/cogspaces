import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import load, Memory, dump, Parallel, delayed
from nilearn._utils import check_niimg
from nilearn.image import iter_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
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


def analyse(output_dir):
    mem = Memory(cachedir=expanduser('~/cache'))

    lr1 = mem.cache(analyse_unsupervised)()

    estimator = load(join(output_dir, 'estimator.pkl'))
    target_encoder = load(join(output_dir, 'target_encoder.pkl'))

    embedder_coef = estimator.module_.embedder.linear.sparse_weight.data.numpy()
    lr2 = DenoisingLinearRegresion()
    snr = np.exp(- .5 * estimator.module_.embedder.linear.log_alpha.data.numpy())
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
    fig = plt.figure()
    plot_stat_map(img, title=name, figure=fig, threshold=0)
    plt.savefig(join(output_dir, '%s.png' % name))
    plt.close(fig)


def plot_maps(lr1, lr2, lr3s, output_dir):
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    # img1 = masker.inverse_transform(lr1.coef_)
    img2 = masker.inverse_transform(lr2.coef_)
    img2_snr = masker.inverse_transform(lr2.snr_)
    img3s = {study: masker.inverse_transform(lr3.coef_) for study, lr3 in lr3s.items()}
    # Parallel(n_jobs=3)(delayed(plot_single)(img, 'level_1_%i.png' % i, output_dir)
    #                    for i, img in enumerate(iter_img(img1)))
    Parallel(n_jobs=3)(delayed(plot_single)(img, 'level_2_%i.png' % i, output_dir)
                       for i, img in enumerate(iter_img(img2)))
    Parallel(n_jobs=3)(delayed(plot_single)(img, 'level_2_%i_snr.png' % i, output_dir)
                       for i, img in enumerate(iter_img(img2_snr)))
    Parallel(n_jobs=3)(delayed(plot_single)(img, 'level_3_%s_%s.png' % (study, name), output_dir)
                       for study, img3 in img3s.items()
                       for name, img in zip(lr3s[study].names_, iter_img(img3)))


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'multi_studies', '1954')

    introspect_dir = join(output_dir, 'introspect')
    plot_dir = join(output_dir, 'introspect', 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    lr1, lr2, lr3s = analyse(output_dir)
    plot_maps(lr1, lr2, lr3s, plot_dir)
