# Baseline logistic
import json
import numpy as np
import os
import pandas as pd
import re
from joblib import load, dump
from os.path import join

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir

idx = pd.IndexSlice


def gather_factored(output_dir, flavor='simple'):
    regex = re.compile(r'[0-9]+$')
    res = []
    confusions = {study: [] for study in get_studies()}
    if flavor == 'refit':
        extra_indices = ['alpha']
    else:
        extra_indices = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
            seed = config['seed']
            test_scores = run['result']

            confusion, _, _, _ = load(join(this_exp_dir, 'scores.pkl'))
            if flavor == 'single_study':
                study = config['data']['studies']
                this_res = dict(study=study, score=test_scores[study],
                                seed=seed)
                res.append(this_res)
                confusions[study].append(confusion[study][:, :, None])
            else:
                for study, score in test_scores.items():
                    study_res = dict(seed=seed, study=study, score=score)
                    if flavor == 'refit':
                        refit_from = config['factored']['refit_from']
                        alpha = float(refit_from[-9:-4])
                        study_res['alpha'] = alpha
                    res.append(study_res)
                    confusions[study].append(confusion[study][:, :, None])
        except:
            print('Skipping exp %i' % this_dir)
            continue
    res = pd.DataFrame(res)
    res = res.set_index(['study', *extra_indices, 'seed'])['score']
    res_mean = res.groupby(['study', *extra_indices]).aggregate(['mean', 'std'])
    print(res_mean)
    res.to_pickle(join(output_dir, 'gathered.pkl'))
    res_mean.to_pickle(join(output_dir, 'gathered_mean.pkl'))

    confusions = {study: np.concatenate(confusion, axis=2).mean(axis=2) if confusion else None
                  for study, confusion in confusions.items()}
    dump(confusions, 'confusion.pkl')


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
    res_mean = res.groupby(['study', 'weight_power']).aggregate(
        ['mean', 'std'])
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


def get_studies():
    source_dir = join(get_data_dir(), 'reduced_512')
    _, target = load_data_from_dir(data_dir=source_dir)
    studies = list(target.keys())
    return studies


if __name__ == '__main__':
    # gather_factored(join(get_output_dir(), 'factored_gm'))
    # gather_factored(join(get_output_dir(), 'factored_refit_gm'), flavor='refit')
    # gather_factored(join(get_output_dir(), 'factored_refit_gm_tune_last'), flavor='refit')
    # gather_factored(join(get_output_dir(), 'old/factored_refit_gm_notune'), flavor='refit')
    # gather_factored(join(get_output_dir(), 'factored_refit_gm'), flavor='refit')
    # gather_factored(join(get_output_dir(), 'factored_refit_gm_notune'), flavor='refit')
    gather_factored(join(get_output_dir(), 'logistic_gm'), flavor='single_study')
    gather_factored(join(get_output_dir(), 'logistic_gm_ds009'), flavor='single_study')
