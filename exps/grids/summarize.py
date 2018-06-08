# Baseline logistic
import json
import os
import pandas as pd
import re
from os.path import join

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir

idx = pd.IndexSlice


def gather_seed_split_init(output_dir):
    regex = re.compile(r'[0-9]+$')
    res = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        model_seed = config['factored']['seed']
        test_scores = run['result']
        this_res = dict(seed=seed, model_seed=model_seed,
                        **test_scores)
        res.append(this_res)
    res = pd.DataFrame(res)
    res = res.set_index(['seed', 'model_seed'])
    studies = res.columns.values
    res = [res[study] for study in studies]
    res = pd.concat(res, keys=studies, names=['study'], axis=0)
    res = res.groupby(['study', 'seed']).aggregate('mean')
    res.sort_index(inplace=True)
    res_mean = res.groupby(['study']).aggregate(['mean', 'std'])
    print(res_mean)
    res.to_pickle(join(output_dir, 'gathered.pkl'))
    res_mean.to_pickle(join(output_dir, 'gathered_mean.pkl'))


def gather_factored_pretrain(output_dir):
    regex = re.compile(r'[0-9]+$')
    res = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        sparsify = config['factored']['max_iter']['sparsify'] > 0
        test_scores = run['result']
        this_res = dict(seed=seed, sparsify=sparsify,
                        **test_scores)
        res.append(this_res)
    res = pd.DataFrame(res)
    res = res.set_index(['sparsify', 'seed'])
    studies = res.columns.values
    res = [res[study] for study in studies]
    res = pd.concat(res, keys=studies, names=['study'], axis=0)
    res.sort_index(inplace=True)
    res_mean = res.groupby(['study', 'sparsify']).aggregate(['mean', 'std'])
    print(res_mean)
    res.to_pickle(join(output_dir, 'gathered.pkl'))
    res.to_pickle(join(output_dir, 'gathered_mean.pkl'))


def gather_init_refit(output_dir):
    regex = re.compile(r'[0-9]+$')
    res = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        init = config['factored']['full_init']
        if 'pca_' in init:
            init = 'pca'
        elif 'dl_rest_' in init:
            init = 'dl_rest'
        elif 'dl_random_' in init:
            init = 'dl_random'
        else:
            raise ValueError
        test_scores = run['result']
        if test_scores is None:
            continue
        this_res = dict(seed=seed, init=init,
                        **test_scores)
        res.append(this_res)
    res = pd.DataFrame(res)
    res = res.set_index(['init', 'seed'])
    studies = res.columns.values
    res = [res[study] for study in studies]
    res = pd.concat(res, keys=studies, names=['study'], axis=0)
    res.sort_index(inplace=True)
    res_mean = res.groupby(['study', 'init']).aggregate(['mean', 'std'])
    print(res_mean)
    res.to_pickle(join(output_dir, 'gathered.pkl'))
    res_mean.to_pickle(join(output_dir, 'gathered_mean.pkl'))


def gather_logistic_refit_l2(output_dir):
    regex = re.compile(r'[0-9]+$')
    res = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        init = config['logistic']['reduction']
        if 'pca_' in init:
            init = 'pca'
        elif 'dl_rest_init_' in init:
            init = 'dl_rest_init'
        elif 'dl_random_init' in init:
            init = 'dl_random_init'
        else:
            raise ValueError
        test_scores = run['result']
        l2_penalty = config['logistic']['l2_penalty']
        this_res = dict(seed=seed, init=init, l2_penalty=l2_penalty,
                        **test_scores)
        res.append(this_res)
    res = pd.DataFrame(res)
    res = res.set_index(['init', 'l2_penalty', 'seed'])
    studies = res.columns.values
    res = [res[study] for study in studies]
    res = pd.concat(res, keys=studies, names=['study'], axis=0)
    res.sort_index(inplace=True)
    res.name = 'score'

    indices = res.groupby(['study', 'init', 'l2_penalty']).aggregate(
        'mean').groupby(['study', 'init']).aggregate('idxmax')
    res_ = []
    keys = []
    for study, init, l2_penalty in indices:
        keys.append((study, init))
        res_.append(res.loc[idx[study, init, l2_penalty, :]])
    res = pd.concat(res_, axis=0)
    res.reset_index('l2_penalty', drop=True, inplace=True)
    res_mean = res.groupby(['study', 'init']).aggregate(['mean', 'std'])
    res.to_pickle(join(output_dir, 'gathered.pkl'))
    res_mean.to_pickle(join(output_dir, 'gathered_mean.pkl'))


def gather_reduced_logistic(output_dir):
    regex = re.compile(r'[0-9]+$')
    res = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        l2_penalty = config['logistic']['l2_penalty']
        study = config['data']['studies']
        score = run['result'][study]
        this_res = dict(seed=seed, l2_penalty=l2_penalty, study=study, score=score)
        res.append(this_res)
    res = pd.DataFrame(res)
    res = res.set_index(['study', 'l2_penalty', 'seed'])

    res = res['score']
    indices = res.groupby(['study', 'l2_penalty']).aggregate(
        'mean').groupby('study').aggregate('idxmax')
    res_ = []
    studies = []
    for study, l2_penalty in indices:
        studies.append(study)
        res_.append(res.loc[idx[study, l2_penalty, :]])
    res = pd.concat(res_, keys=studies, names=['study'], axis=0)
    res.sort_index(inplace=True)
    res_mean = res.groupby(['study']).aggregate(['mean', 'std'])
    print(res_mean)
    res.to_pickle(join(output_dir, 'gathered.pkl'))
    res_mean.to_pickle(join(output_dir, 'gathered_mean.pkl'))


def gather_single_factored(output_dir):
    regex = re.compile(r'[0-9]+$')
    res = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        study = config['data']['studies']
        score = run['result'][study]
        this_res = dict(seed=seed, study=study, score=score)
        res.append(this_res)
    res = pd.DataFrame(res)
    res = res.set_index(['study', 'seed'])

    res.sort_index(inplace=True)
    res_mean = res.groupby(['study']).aggregate(['mean', 'std'])
    print(res_mean)
    res.to_pickle(join(output_dir, 'gathered.pkl'))
    res_mean.to_pickle(join(output_dir, 'gathered_mean.pkl'))



def gather_dropout(output_dir):
    regex = re.compile(r'[0-9]+$')
    res = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        study = config['data']['studies']
        dropout = config['factored']['dropout']
        adaptive_dropout = config['factored']['adaptive_dropout']
        test_scores = run['result']
        this_res = dict(seed=seed, dropout=dropout,
                        adaptive_dropout=adaptive_dropout,
                        **test_scores)
        res.append(this_res)
    res = pd.DataFrame(res)
    res = res.set_index(['adaptive_dropout', 'dropout', 'seed'])
    studies = res.columns.values
    res = [res[study] for study in studies]
    res = pd.concat(res, keys=studies, names=['study'], axis=0)
    res.sort_index(inplace=True)
    res_mean = res.groupby(['study', 'adaptive_dropout',
                       'dropout']).aggregate(['mean', 'std'])
    print(res_mean)
    res.to_pickle(join(output_dir, 'gathered.pkl'))
    res_mean.to_pickle(join(output_dir, 'gathered_mean.pkl'))


def gather_weight_power(output_dir):
    regex = re.compile(r'[0-9]+$')
    res = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        seed = config['seed']
        study = config['data']['studies']
        gather_weight_power = config['model']['gather_weight_power']
        score = run['result'][study]
        this_res = dict(seed=seed,
                        gather_weight_power=gather_weight_power,
                        study=study, score=score)
        res.append(this_res)
    res = pd.DataFrame(res)
    res = res.set_index(['study', 'weight_power', 'seed'])
    res.sort_index(inplace=True)
    res_mean = res.groupby(['study', 'weight_power']).aggregate(['mean', 'std'])
    print(res_mean)
    res.to_pickle(join(output_dir, 'gathered.pkl'))
    res_mean.to_pickle(join(output_dir, 'gathered_mean.pkl'))


def get_chance_subjects():
    _, target = load_data_from_dir(
        data_dir=join(get_data_dir(), 'reduced_512'))
    chance_level = {}
    n_subjects = {}
    for study, this_target in target.items():
        chance_level[study] = 1. / len(this_target['contrast'].unique())
        n_subjects[study] = len(this_target['subject'].unique())

    chance_level = pd.Series(chance_level)
    n_subjects = pd.Series(n_subjects)
    return chance_level, n_subjects

    
def join_baseline_factored(baseline_output_dir, factored_output_dir):
    factored = pd.read_pickle(join(factored_output_dir,
                                   'gathered.pkl'))
    baseline = pd.read_pickle(join(baseline_output_dir, 'gathered.pkl'))

    chance_level, n_subjects = get_chance_subjects()

    joined = pd.concat([factored, baseline, chance_level, n_subjects],
                       keys=['factored', 'baseline'], axis=1)
    joined['diff'] = joined['factored'] - joined['baseline']
    joined_mean = joined.groupby('study').aggregate(['mean', 'std'])
    joined_mean['chance'] = chance_level
    joined_mean['n_subjects'] = n_subjects

    joined.to_pickle(join(get_output_dir(), 'joined.pkl'))
    joined_mean.to_pickle(join(get_output_dir(), 'joined_mean.pkl'))


if __name__ == '__main__':
    # gather_seed_split_init(join(get_output_dir(), 'seed_split_init'))
    # gather_reduced_logistic(join(get_output_dir(), 'reduced_logistic'))
    # gather_dropout(join(get_output_dir(), 'dropout'))
    # gather_single_factored(join(get_output_dir(), 'single_factored'))
    # gather_init_refit(join(get_output_dir(), 'init_refit_dense'))
    gather_init_refit(join(get_output_dir(), 'init_refit_finetune'))
    # gather_factored_pretrain(join(get_output_dir(), 'factored_pretrain'))
    # gather_logistic_refit_l2(join(get_output_dir(), 'logistic_refit_l2'))
    # gather_weight_power(join(get_output_dir(), 'gather_weight_power'))

    # join_baseline_factored(join(get_output_dir(), 'reduced_logistic'),
    #                        join(get_output_dir(), 'seed_split_init'))