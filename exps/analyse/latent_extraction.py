import time

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from flask import json
from joblib import Parallel, delayed, load
from modl.decomposition.dict_fact import DictFact
from nilearn._utils import check_niimg
from nilearn.image import iter_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import find_xyz_cut_coords, plot_glass_brain
from numpy.linalg import qr
from os.path import join
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir


def explained_variance(X, components, per_component=True):
    """Score function based on explained variance

        Parameters
        ----------
        data: ndarray,
            Holds single subject data to be tested against components

        per_component: boolean,
            Specify whether the explained variance ratio is desired for each
            map or for the global set of components_

        Returns
        -------
        score: ndarray,
            Holds the score for each subjects. score is two dimensional if
            per_component = True
        """
    full_var = np.var(X)
    n_components = components.shape[0]
    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components = components / S[:, np.newaxis]
    projected_data = X.dot(components.T)
    if per_component:
        res_var = np.zeros(n_components)
        for i in range(n_components):
            res = X - projected_data[:, i][:, None] * components[i][None, :]
            res_var[i] = np.var(res)
        return np.maximum(0., 1. - res_var / full_var)
    else:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(components.T, X.T)
        residuals = X - lr.coef_.dot(components)
        res_var = np.var(residuals)
        return np.maximum(0., 1. - res_var / full_var)


class DictionaryScorer:
    def __init__(self, test_data):
        self.start_time = time.clock()
        self.test_data = test_data
        self.test_time = 0
        self.time = []
        self.cpu_time = []
        self.score = []
        self.iter = []
        self.density = []
        self.exp_var = []

    def __call__(self, dict_fact):
        test_time = time.clock()
        score = dict_fact.score(self.test_data)

        exp_var = explained_variance(self.test_data, dict_fact.components_,
                                     per_component=False)
        self.exp_var.append(exp_var)

        self.test_time += time.clock() - test_time
        this_time = time.clock() - self.start_time - self.test_time
        self.time.append(this_time)
        self.score.append(score)
        self.iter.append(dict_fact.n_iter_)
        self.cpu_time.append(dict_fact.time_)
        self.density.append((dict_fact.components_ != 0).mean())
        print(self.density[-1])


def compute_coefs(output_dir):
    regex = re.compile(r'[0-9]+$')
    res = []
    dropout = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            info = json.load(
                open(join(this_exp_dir, 'info.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        model_seed = config['factored']['seed']
        dropout.append(dict(seed=seed, **info['dropout'], model_seed=model_seed))
        this_res = dict(seed=seed, dir=this_dir)
        res.append(this_res)
    res = pd.DataFrame(res)
    dropout = pd.DataFrame(dropout)
    res.to_pickle(join(output_dir, 'seeds.pkl'))
    for seed, sub_res in res.groupby(by='seed'):
        print(seed)
        all_coefs = []
        for this_dir in sub_res['dir']:
            estimator = load(join(output_dir, str(this_dir),
                                  'estimator.pkl'))
            coef = estimator.module_.embedder.linear.weight.detach().numpy()
            all_coefs.append(coef)
        all_coefs = np.concatenate(all_coefs, axis=0)
        np.save(join(output_dir, 'coefs_%i.npy' % seed), all_coefs)
    dropout = dropout.set_index(['seed', 'model_seed'])
    dropout = dropout.groupby('seed').mean()
    dropout.to_pickle(join(output_dir, 'dropout.pkl'))

def fetch_atlas_and_masker():
    modl_atlas = fetch_atlas_modl()
    atlas = check_niimg(modl_atlas['components512'])
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    atlas = masker.transform(atlas)
    return atlas, masker


def compute_pca(output_dir, seed):
    coefs = np.load(join(output_dir, 'coefs_%i.npy' % seed))

    pca = PCA(n_components=128)
    pca = pca.fit(coefs)
    components = pca.components_
    exp_vars = pca.explained_variance_ratio_

    explained_var = np.sum(exp_vars)
    print('Total explained variance %.4f' % explained_var)

    return components


def compute_sparse_components(output_dir, seed, init='rest',
                              symmetric_init=True,
                              proj_principal=False,
                              alpha=1e-2):
    coefs = np.load(join(output_dir, 'coefs_%i.npy' % seed))

    if init == 'random':
        random_state = check_random_state(0)
        dict_init = random_state.randn(512, 128)
        q, r = qr(dict_init)
        q = q * np.sign(np.diag(r))
        dict_init = q.T
        dict_init /= np.sqrt(512)
    elif init == 'rest':
        loadings_128 = fetch_atlas_modl()['loadings128']
        dict_init = np.load(loadings_128)
    elif init == 'data':
        random_state = check_random_state(0)
        indices = random_state.permutation(len(coefs))[:128]
        dict_init = coefs[indices]
    if symmetric_init:
        assign = fetch_atlas_modl()['assign512']
        assign = np.load(assign)
        dict_init += dict_init[:, assign]
        dict_init /= 2

    sc = StandardScaler(with_std=False, with_mean=True)
    sc.fit(coefs)
    coefs_ = sc.transform(coefs)

    if proj_principal:
        pca = PCA(n_components=128)
        pca = pca.fit(coefs_)
        dict_init = pca.inverse_transform(pca.transform(dict_init))
        coefs_ = pca.inverse_transform(pca.transform(coefs_))

    # if init == 'rest':
    #     dict_fact = DictFact(comp_l1_ratio=0, comp_pos=False,
    #                          n_components=128,
    #                          code_l1_ratio=0, batch_size=32,
    #                          learning_rate=1,
    #                          dict_init=dict_init,
    #                          code_alpha=alpha, verbose=0, n_epochs=3,
    #                          )
    #     dict_fact.fit(coefs_)
    #     dict_init = dict_fact.components_
    dict_fact = DictFact(comp_l1_ratio=0, comp_pos=False, n_components=128,
                         code_l1_ratio=0, batch_size=32, learning_rate=1,
                         dict_init=dict_init,
                         code_alpha=alpha, verbose=10, n_epochs=20,
                         )
    dict_fact.fit(coefs_)

    components = dict_fact.components_

    total_exp_var = explained_variance(coefs, components,
                                       per_component=False)
    print('Total exp var: %.4f' % total_exp_var)
    density = (components != 0).mean()
    print('Density: %.4f' % density)

    exp_vars = explained_variance(coefs[:500], components,
                                  per_component=True)
    sort = np.argsort(exp_vars)[::-1]
    components = components[sort]
    exp_vars = exp_vars[sort]

    return components


def plot_single(img, title, filename):
    vmax = np.abs(img.get_data()).max()
    cut_coords = find_xyz_cut_coords(img, activation_threshold=vmax / 3)
    fig = plt.figure()
    plot_glass_brain(img, cut_coords=cut_coords, threshold=0,
                     figure=fig, title=title,
                     plot_abs=False, vmax=vmax)
    plt.savefig(filename)
    plt.close(fig)


def plot_all(imgs, exp_vars, dest_dir, n_jobs=1):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    imgs = check_niimg(imgs)
    imgs.get_data()
    Parallel(n_jobs=n_jobs)(delayed(plot_single)
                            (img, 'exp var %.4f' % exp_var,
                             join(dest_dir, 'components_%i.png' % i))
                            for i, (img, exp_var) in
                            enumerate(zip(iter_img(imgs),
                                          exp_vars)))


def compute_all_decomposition(output_dir):
    seeds = pd.read_pickle(join(output_dir, 'seeds.pkl'))
    seeds = seeds['seed'].unique()

    # components_list = Parallel(n_jobs=20, verbose=10)(
    #     delayed(compute_pca)(output_dir, seed)
    #     for seed in seeds)
    # for components, seed in zip(components_list, seeds):
    #     np.save(join(output_dir, 'pca_%i.npy' % seed), components)

    components_list = Parallel(n_jobs=20, verbose=10)(
        delayed(compute_sparse_components)
        (output_dir, seed,
         symmetric_init=False,
         alpha=1e-4,
         init='rest')
        for seed in seeds)
    for components, seed in zip(components_list, seeds):
        np.save(join(output_dir, 'dl_rest_init_dense_%i.npy' % seed), components)

    components_list = Parallel(n_jobs=20, verbose=10)(
        delayed(compute_sparse_components)
        (output_dir, seed,
         symmetric_init=False,
         alpha=1e-4,
         init='random')
        for seed in seeds)
    for components, seed in zip(components_list, seeds):
        np.save(join(output_dir, 'dl_random_init_dense_%i.npy' % seed), components)


#
# def compute_components_and_plot(output_dir, init='rest', symmetric_init=True,
#                                 proj_principal=False, alpha=1e-2):
#     mem = Memory(get_cache_dir())
#     coefs = np.load(join(output_dir, 'coefs.npy'))
#
#     components = compute_components(coefs, init, symmetric_init,
#                                     proj_principal, alpha=alpha)
#     density = (components != 0).mean()
#     exp_var = explained_variance(coefs, components,
#                                  per_component=False)
#     print('Density:', density)
#     print('Explained variance:', exp_var)
#
#     np.save(join(output_dir, 'dl_%s.npy' % init), components)
#
#     exp_vars = explained_variance(coefs[:500], components,
#                                   per_component=True)
#     sort = np.argsort(exp_vars)[::-1]
#     components = components[sort]
#     exp_vars = exp_vars[sort]
#     print(exp_vars)
#
#
#     atlas, masker = mem.cache(fetch_atlas_and_masker)()
#
#     maps = components.dot(atlas)
#
#     maps = masker.inverse_transform(maps)
#
#     maps.to_filename(join(output_dir, 'dl_%s.nii.gz' % init))
#     dest_dir = join(output_dir, 'dl_%s' % init)
#     plot_all(maps, exp_vars, dest_dir, n_jobs=3)


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'seed_split_init')
    # compute_coefs(output_dir)
    compute_all_decomposition(output_dir)
