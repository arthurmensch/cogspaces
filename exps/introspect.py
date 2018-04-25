import json
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import load, delayed, Parallel, Memory
from modl import DictFact
from nilearn._utils import check_niimg
from nilearn.image import index_img, iter_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map, plot_prob_atlas, \
    find_xyz_cut_coords
from os.path import join, expanduser
from scipy.linalg import svd
from torch.utils.data import TensorDataset

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir
from cogspaces.model_selection import train_test_split
from cogspaces.models.factored_fast import MultiStudyLoader
from cogspaces.preprocessing import MultiTargetEncoder
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


def compute_latent(output_dir):
    introspect_dir = join(output_dir, 'introspect')
    if not os.path.exists(introspect_dir):
        os.makedirs(introspect_dir)
    estimator = load(join(output_dir, 'estimator.pkl'))

    config = json.load(open(join(output_dir, 'config.json'), 'r'))
    lstsq = config['data']['source_dir'] == 'reduced_512_lstsq'
    print(lstsq)

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
    weight = estimator.module_sparsify_.embedder.linear.weight.data.numpy()
    snr = - .5 * estimator.module_sparsify_.embedder.linear.log_alpha.data.numpy()
    snr = np.exp(snr)
    proj = weight @ dict_proj
    proj_snr = snr @ dict_proj
    weighted_proj = proj * proj_snr
    img = masker.inverse_transform(proj)
    img_snr = masker.inverse_transform(proj_snr)
    img_weighted = masker.inverse_transform(weighted_proj)
    img.to_filename(join(introspect_dir, 'components.nii.gz'))
    img_snr.to_filename(join(introspect_dir, 'components_snr.nii.gz'))
    img_weighted.to_filename(join(introspect_dir, 'components_weighted.nii.gz'))

    # target_encoder = load(join(output_dir, 'target_encoder.pkl'))
    #
    # for study, classifier in estimator.module_.classifiers.items():
    #     weight = classifier.linear.weight.data.numpy()
    #     these_weights = weight @ proj
    #     these_weights -= these_weights.mean(axis=0)[None, :]
    #     classification_maps = masker.inverse_transform(these_weights)
    #     these_names = target_encoder.le_[study]['contrast'].classes_
    #     classification_maps.to_filename(join(introspect_dir,
    #                                          'classification_%s.nii.gz'
    #                                          % study))
    #     dump(these_names, join(introspect_dir,
    #                            'classification_%s-names.pkl' % study))
    #     with open(join(introspect_dir, 'classification_%s-names' % study),
    #               'w+') as f:
    #         f.write(str(these_names))


def plot_latent(output_dir):
    introspect_dir = join(output_dir, 'introspect')
    plot_dir = join(introspect_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    img = check_niimg(join(introspect_dir, 'components.nii.gz'))
    img_snr = check_niimg(join(introspect_dir, 'components_snr.nii.gz'))
    img_weighted = check_niimg(join(introspect_dir, 'components_weighted.nii.gz'))
    plot_prob_atlas(img)
    plt.savefig(join(plot_dir, 'components.png'))
    plt.close()
    Parallel(n_jobs=20)(delayed(plot_single_img)(
        str(i), plot_dir, 'components', this_img)
                        for (i, this_img)
                        in enumerate(iter_img(img)))
    Parallel(n_jobs=20)(delayed(plot_single_img)(
        str(i), plot_dir, 'snr', this_img)
                        for (i, this_img)
                        in enumerate(iter_img(img_snr)))
    Parallel(n_jobs=20)(delayed(plot_single_img)(
        str(i), plot_dir, 'componentsxsnr', this_img, to=0)
                        for (i, this_img)
                        in enumerate(iter_img(img_weighted)))



def plot_classification(output_dir):
    introspect_dir = join(output_dir, 'introspect')
    plot_dir = join(introspect_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    pattern = re.compile(r'classification_(?P<study>\w*).nii.gz')
    for this_file in os.listdir(introspect_dir):
        match = re.match(pattern, this_file)
        if match is not None:
            study = match['study']
            print(study)
            maps = check_niimg(join(introspect_dir,
                                    'classification_%s.nii.gz' % study))
            names = load(join(introspect_dir, 'classification_%s-names.pkl'
                              % study))
            Parallel(n_jobs=20)(delayed(plot_single_img)(
                name, plot_dir, study, this_img)
                               for (this_img, name)
                               in zip(iter_img(maps), names))


def plot_single_img(name, plot_dir, study, this_img, to=1/6):
    vmax = np.max(np.abs(this_img.get_data()))
    cut_coords = find_xyz_cut_coords(this_img,
                                     activation_threshold=vmax / 3)
    plt.figure()
    plot_stat_map(this_img, title='%s::%s' % (study, name),
                  cut_coords=cut_coords,
                  threshold=vmax * to)
    plt.savefig(join(plot_dir, '%s_%s.png' % (study, name)))
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


def inspect_latent(output_dir):
    introspect_dir = join(output_dir, 'introspect')
    if not os.path.exists(introspect_dir):
        os.makedirs(introspect_dir)
    estimator = load(join(output_dir, 'estimator.pkl'))

    config = json.load(open(join(output_dir, 'config.json'), 'r'))

    data, target = load_data(config['data']['source_dir'], config['data']['studies'],
                     config['data']['target_study'])
    target_encoder = MultiTargetEncoder().fit(target)
    target = target_encoder.transform(target)

    train_data, test_data, train_targets, test_targets = \
        train_test_split(data, target, random_state=844704211)
    lstsq = True

    # PCA
    # weights = np.array([math.sqrt(latent.shape[0])
    #                     for latent in latents.values()])
    # weights /= np.sum(weights)
    # weights = np.concatenate(list(map(lambda x: np.ones(x.shape[0])
    #                                             * 1 / math.sqrt(x.shape[0]),
    #                                   iter(latents.values()))))
    # X = np.concatenate(list(latents.values()), axis=0)
    # mean = np.sum(X * weights[:, None], axis=0) / np.sum(weights)
    # Xc = X - mean[None, :]
    # cov = np.dot((Xc * weights[:, None]).T, Xc)
    # cov /= np.sum(weights)
    # V, S, Vt = np.linalg.svd(cov)
    # print(Vt.shape)
    # plt.plot(range(S.shape[0]), S)
    # plt.show()

    latents = estimator.predict_latent(train_data)
    X, y = latents, train_targets
    X = {study: torch.from_numpy(this_X).double()
         for study, this_X in X.items()}
    y = {study: torch.from_numpy(this_y['contrast'].values).long()
         for study, this_y in y.items()}
    data = {study: TensorDataset(X[study], y[study]) for study in X}
    print(len)

    study_weights = {study: math.sqrt(len(this_data)) for study, this_data
                     in data.items()}
    print(study_weights)
    for study, this_data in data.items():
        print(study, len(this_data))
    data_loader = MultiStudyLoader(data, sampling='random',
                                   batch_size=128,
                                   seed=0,
                                   study_weights=study_weights,
                                   cuda=False, device=-1)

    dl = DictFact(n_components=128, code_l1_ratio=0, comp_l1_ratio=1,
                  code_alpha=10)
    dl.prepare(n_samples=128, n_features=256)

    i = 0
    for inputs, targets in data_loader:
        for study, input in inputs.items():
            print('%s iter % i' % (study, i))
            dl.partial_fit(input.data.numpy(), sample_indices=np.arange(len(input)))
            i += 1
            if i > 1000:
                break
        if i > 1000:
            break
    Vt = dl.components_

    mem = Memory(cachedir=expanduser('~/cachedir'))
    proj, masker = mem.cache(get_proj_and_masker)(estimator, lstsq)
    principal = Vt @ proj
    img = masker.inverse_transform(principal)
    img.to_filename(join(introspect_dir, 'principal_components.nii.gz'))


def get_proj_and_masker(estimator, lstsq):
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
    return proj, masker


if __name__ == '__main__':
    compute_latent(join(get_output_dir(), 'multi_studies', '418'))
    # plot_latent(join(get_output_dir(), 'multi_studies', '418'))
    # plot_classification(join(get_output_dir(), 'multi_studies', '418'))

    # compute_latent(join(get_output_dir(), 'multi_studies', '413'))
    # plot_latent(join(get_output_dir(), 'multi_studies', '413'))
    # plot_classification(join(get_output_dir(), 'multi_studies', '413'))

    # inspect_latent(join(get_output_dir(), 'multi_studies', '1745'))

    # compute_latent(join(get_output_dir(), 'multi_studies', '1786'))
