import time

import json
import numpy as np
import os
import pandas as pd
import re
import torch
from joblib import Parallel, delayed, load, dump
from modl.decomposition.dict_fact import DictFact
from nilearn._utils import check_niimg
from nilearn.input_data import NiftiMasker
from numpy.linalg import qr, lstsq
from os.path import join
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir
from cogspaces.models.factored_dl import explained_variance
from cogspaces.utils import get_dictionary, get_masker


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
            this_dropout = info['dropout']
        except (FileNotFoundError, json.decoder.JSONDecodeError, KeyError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        model_seed = config['factored']['seed']
        dropout.append(dict(seed=seed, **this_dropout,
                            model_seed=model_seed))
        this_res = dict(seed=seed, dir=this_dir)
        res.append(this_res)
    res = pd.DataFrame(res)
    dropout = pd.DataFrame(dropout)
    res.to_pickle(join(output_dir, 'seeds.pkl'))

    dropout = dropout.set_index(['seed', 'model_seed'])
    dropout = dropout.groupby('seed').mean()

    for seed, sub_res in res.groupby(by='seed'):
        print('Gathering coef for seed', seed)
        seed_dropout = dropout.loc[seed].to_dict()
        studies = seed_dropout.keys()
        n_runs = 0
        latent_coefs = []
        full_coefs = {study: 0 for study in studies}
        full_biases = {study: 0 for study in studies}
        for this_dir in sub_res['dir']:
            try:
                estimator = load(join(output_dir, str(this_dir),
                                      'estimator.pkl'))
                module = estimator.module_
            except FileNotFoundError:
                print('Skipping exp %i' % this_dir)
                continue
            latent_coef = module.embedder.linear.weight.detach().numpy()
            latent_coefs.append(latent_coef)
            in_features = module.embedder.linear.in_features
            module.eval()
            with torch.no_grad():
                full_bias = module({study: torch.zeros((1, in_features))
                                    for study in studies}, logits=True)
                full_coef = module({study: torch.eye(in_features)
                                    for study in studies}, logits=True)
                full_coef = {study: full_coef[study] - full_bias[study]
                             for study in studies}
            for study in studies:
                full_coefs[study] += full_coef[study]
                full_biases[study] += full_bias[study]
            n_runs += 1
        full_coefs = {study: coef.numpy().T / n_runs for
                      study, coef in full_coefs.items()}
        full_biases = {study: bias.numpy()[0] / n_runs for
                          study, bias in full_biases.items()}
        latent_coefs = np.concatenate(latent_coefs, axis=0)
        dump((latent_coefs, full_coefs, full_biases, seed_dropout),
             join(output_dir, 'combined_models_%i.pkl' % seed))


def fetch_atlas_and_masker():
    modl_atlas = fetch_atlas_modl()
    atlas = check_niimg(modl_atlas['components512'])
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    atlas = masker.transform(atlas)
    return atlas, masker


def compute_pca(output_dir, seed):
    coefs, _, _, _ = load(
        join(output_dir, 'combined_models_%s.pkl' % seed))

    pca = PCA(n_components=128)
    pca = pca.fit(coefs)
    components = pca.components_
    exp_vars = pca.explained_variance_ratio_

    explained_var = np.sum(exp_vars)
    print('Total explained variance %.4f' % explained_var)

    return components


def compute_sparse_components(output_dir, seed, init='rest',
                              symmetric_init=True,
                              positive=True,
                              proj_principal=False,
                              alpha=1e-2):
    coefs, _, _, _ = load(
        join(output_dir, 'combined_models_%s.pkl' % seed))
    n_features = coefs.shape[1]
    if init == 'random':
        random_state = check_random_state(0)
        dict_init = random_state.randn(n_features, 128)
        q, r = qr(dict_init)
        q = q * np.sign(np.diag(r))
        dict_init = q.T
        dict_init /= np.sqrt(n_features)
    elif init == 'rest':
        loadings_128 = fetch_atlas_modl()['loadings128_gm']
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

    if init == 'rest':
        dict_fact = DictFact(comp_l1_ratio=0, comp_pos=positive,
                             n_components=128,
                             code_l1_ratio=0, batch_size=32,
                             learning_rate=1,
                             dict_init=dict_init,
                             code_alpha=alpha, verbose=0, n_epochs=2,
                             )
        dict_fact.fit(coefs_)
        dict_init = dict_fact.components_
    dict_fact = DictFact(comp_l1_ratio=1, comp_pos=positive, n_components=128,
                         code_l1_ratio=0, batch_size=32, learning_rate=1,
                         dict_init=dict_init,
                         code_alpha=alpha, verbose=10, n_epochs=20)
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


def compute_all_decomposition(output_dir, n_jobs=1):
    seeds = pd.read_pickle(join(output_dir, 'seeds.pkl'))
    seeds = seeds['seed'].unique()

    decompositions = ['dl_positive']
    # alphas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    alphas = [1e-5, 1e-6, 1e-7]

    for decomposition in decompositions:
        if decomposition == 'pca':
            components_list = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(compute_pca)(output_dir, seed)
                for seed in seeds)
        elif decomposition == 'dl_rest':
            components_list = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(compute_sparse_components)
                (output_dir, seed,
                 symmetric_init=False,
                 alpha=alpha,
                 init='rest')
                for seed in seeds
                for alpha in alphas)
        elif decomposition == 'dl_rest_positive':
            components_list = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(compute_sparse_components)
                (output_dir, seed,
                 symmetric_init=False,
                 positive=True,
                 alpha=alpha,
                 init='rest')
                for seed in seeds
                for alpha in alphas)
        elif decomposition == 'dl_random':
            components_list = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(compute_sparse_components)
                (output_dir, seed,
                 symmetric_init=False,
                 alpha=alpha,
                 init='random')
                for seed in seeds
                for alpha in alphas)
        elif decomposition == 'dl_positive':
            components_list = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(compute_sparse_components)
                (output_dir, seed,
                 positive=True,
                 symmetric_init=False,
                 alpha=alpha,
                 init='random')
                for seed in seeds
                for alpha in alphas)
        for seed in seeds:
            for alpha in alphas:
                components = components_list[0]
                components_list = components_list[1:]
                (latent_coefs, full_coefs, full_biases, dropout) = load(
                    join(output_dir, 'combined_models_%s.pkl' % seed))
                classif_coefs = {}
                for study in full_coefs:
                    classif_coef, _, _, _ = lstsq(components.T,
                                                  full_coefs[study].T,
                                                  rcond=None)
                    classif_coefs[study] = classif_coef.T
                classif_biases = full_biases
                dump((components, classif_coefs, classif_biases, dropout),
                     join(output_dir, '%s_%i_%.0e.pkl' % (decomposition,
                                                          seed, alpha)))


def nifti_all(output_dir):
    seeds = pd.read_pickle(join(output_dir, 'seeds.pkl'))
    seeds = seeds['seed'].unique()

    dictionary = get_dictionary()
    masker = get_masker()

    for decomposition in ['dl_rest']:
        for seed in seeds:
            name = '%s_%i' % (decomposition, seed)
            (components, _, _, _) = load(join(output_dir, '%s.pkl' % name))
            components = components.dot(dictionary)
            components = masker.inverse_transform(components)
            components.to_filename(join(output_dir, '%s.nii.gz' % name))


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'factored_gm_normal_init_full')
    # output_dir = join(get_output_dir(), 'factored_gm_normal_init')
    compute_coefs(output_dir)
    compute_all_decomposition(output_dir, n_jobs=40)
    # nifti_all(output_dir)
