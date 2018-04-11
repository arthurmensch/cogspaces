import os

import matplotlib.pyplot as plt
import numpy as np
from joblib import load, Memory
from nilearn._utils import check_niimg
from nilearn.decomposition import CanICA
from nilearn.image import index_img, iter_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map, plot_prob_atlas
from os.path import join, expanduser
from scipy.linalg import svd
from sklearn.decomposition import PCA, FastICA, fastica, dict_learning_online
from sklearn.utils.extmath import randomized_svd

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_data_dir
from exps.train import load_data


def plot_components(components, names, output_dir):
    for study in components:
        this_img = components[study]
        these_names = names[study]

        for i, name in enumerate(these_names):
            full_name = study + ' ' + name
            fig = plt.figure()
            plot_stat_map(index_img(this_img, i), figure=fig, title=full_name)
            plt.savefig(join(output_dir, '%s.png' % full_name))
            plt.close(fig)


def compute_latent(output_dir, lstsq):
    estimator = load(join(output_dir, 'estimator.pkl'))

    modl_atlas = fetch_atlas_modl()
    dictionary = modl_atlas['components512']
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    dictionary = masker.transform(dictionary)
    if lstsq:
        gram = dictionary.dot(dictionary.T)
        dict_proj = np.linalg.inv(gram).dot(dictionary)
    else:
        dict_proj = dictionary
    sup_proj = estimator.module_.embedder.linear.weight.data.numpy()
    proj = sup_proj @ dict_proj

    memory = Memory(cachedir=expanduser('~/cache'))

    components, variance, _ = memory.cache(randomized_svd)(proj.T,
                                                           n_components=128)
    _, _, sources = memory.cache(fastica)(components, whiten=True, fun='cube')
    dict_init, _, _, _ = memory.cache(np.linalg.lstsq)(sources, proj.T)

    code, dictionary = memory.cache(dict_learning_online)(proj.T,
                                                          n_components=128,
                                                          alpha=.1,
                                                          batch_size=32,
                                                          dict_init=dict_init,
                                                          return_code=True,
                                                          method='cd',
                                                          verbose=10,
                                                          n_iter=proj.shape[
                                                                     1] // 32)
    S = code.sum(axis=0) < 0
    code[:, S] *= -1
    img = masker.inverse_transform(code.T)
    img.to_filename(expanduser('~/components_orth.nii.gz'))

    gram = proj @ proj.T
    back_proj = np.linalg.inv(gram) @ proj

    # components, variance, _ = randomized_svd(proj.T, n_components=128)
    # _, _, sources = fastica(components, whiten=True, fun='cube')
    # img = masker.inverse_transform(sources.T)
    # img.to_filename(expanduser('~/components_orth.nii.gz'))

    # source_dir = join(get_data_dir(), 'reduced_512_lstsq')
    # data, target = load_data(source_dir, 'all', 'archi')
    # data = {'archi': data['archi']}
    # latents = estimator.predict_latent(data)['archi']
    # rec = latents.dot(back_proj)
    # print(data)
    # print(rec)

    # print(projector.shape)
    # fast_ica = FastICA(whiten=False, algorithm='deflation')
    # fast_ica.fit(projector)
    # projector = fast_ica.components_
    # img = masker.inverse_transform(projector)
    # img.to_filename(expanduser('~/components.nii.gz'))


#
#
# def compute_components(output_dir, lstsq):
#     estimator = load(join(output_dir, 'estimator.pkl'))
#     target_encoder = load(join(output_dir, 'target_encoder.pkl'))
#     standard_scaler = load(join(output_dir, 'standard_scaler.pkl'))
#
#     modl_atlas = fetch_atlas_modl()
#     dictionary = modl_atlas['components512']
#     plot_dir = join(output_dir, 'plot')
#     if not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)
#     dump(names, join(plot_dir, 'names.pkl'))
#     for study, this_components in components.items():
#         this_components.to_filename(join(plot_dir,
#                                          'components_%s.nii.gz' % study))
#         plot_components(components, names, plot_dir)
#
#


def plot_components():
    img = check_niimg(expanduser('~/components_orth.nii.gz'))
    output_dir = expanduser('~/components_orth')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_prob_atlas(img)
    plt.savefig(join(output_dir, 'atlas.png'))
    plt.close()
    for i in range(128):
        plot_stat_map(index_img(img, i))
        plt.savefig(join(output_dir, 'components_%i.png' % i))
        plt.close()


def plot_activation(output_dir):
    test_latents = load(join(output_dir, 'test_latents.pkl'))
    train_latents = load(join(output_dir, 'train_latents.pkl'))
    test_latent_all = np.concatenate(list(test_latents.values()))
    train_latent_all = np.concatenate(list(train_latents.values()))

    U, s_train, Vh = svd(test_latent_all)
    U, s_test, Vh = svd(train_latent_all)

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(s_test)), s_test)
    ax.plot(range(len(s_train)), s_train)
    plt.savefig(join(output_dir, 'all_svd.png'))
    plt.close(fig)

    for study in test_latents:
        test_latent = test_latents[study]
        train_latent = train_latents[study]
        test_latent -= test_latent.mean(axis=0)
        train_latent -= train_latent.mean(axis=0)
        test_latent_std = test_latent.std(axis=0)
        train_latent_std = train_latent.std(axis=0)

        U, s_train, Vh = svd(train_latent)
        U, s_test, Vh = svd(test_latent)

        fig, ax = plt.subplots(1, 1)
        ax.plot(range(len(s_test)), s_test)
        ax.plot(range(len(s_train)), s_train)
        plt.savefig(join(output_dir, '%s_svd.png' % study))
        plt.close(fig)

        dim = test_latent_std.shape[0]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.bar(range(dim), test_latent_std, width=1)
        ax2.bar(range(dim), train_latent_std, width=1)
        ax1.set_title('Test data')
        ax2.set_title('train data')
        fig.suptitle(study)
        for ax in (ax1, ax2):
            ax.set_xlim([0, dim])
            ax.set_ylim([0, 5])
            ax.set_xlabel('Channel')
            ax.set_ylabel('Mean level')
        plt.savefig(join(output_dir, '%s_latent.png' % study))
        plt.close(fig)


if __name__ == '__main__':
    compute_latent(expanduser('~/322'), True)
    plot_components()
    # compute_components(join(get_output_dir(), 'multi_studies', '107'),
    #                    lstsq=True)
    # plot_activation(join(get_output_dir(), 'multi_studies', '922'))
