# Baseline logistic
from math import ceil

import json
import numpy as np
import os
import pandas as pd
import re
from joblib import load
from os.path import join

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir

idx = pd.IndexSlice


def gather_factored(output_dir, flavor='simple'):
    regex = re.compile(r'[0-9]+$')
    if flavor == 'refit':
        extra_indices = ['alpha']
    else:
        extra_indices = []
    accuracies = []
    metrics = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
            seed = config['seed']
            test_scores = run['result']
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue

        Cs, precs, f1s, recalls = load(
            join(this_exp_dir, 'scores.pkl'))
        baccs = {}
        for study in precs:
            C = Cs[study]
            total = np.sum(C)
            baccs[study] = {}
            for i, contrast in enumerate(precs[study]):
                t = np.sum(C[i])
                p = np.sum(C[:, i])
                n = total - p
                tp = C[i, i]
                fp = p - tp
                fn = t - tp
                tn = total - fp - fn - tp
                baccs[study][contrast] = .5 * (tp / p + tn / n)

        if flavor == 'single_study':
            study = config['data']['studies']
            accuracies.append(dict(study=study, accuracy=test_scores[study],
                                   seed=seed))
            f1s = f1s[study]
            recalls = recalls[study]
            precs = precs[study]
            baccs = baccs[study]

            metrics.extend([dict(study=study, contrast=contrast,
                                 prec=precs[contrast],
                                 recall=recalls[contrast],
                                 bacc=baccs[contrast],
                                 f1=f1s[contrast],
                                 seed=seed)
                            for contrast in f1s])
        else:
            if flavor == 'refit':
                refit_from = config['factored']['refit_from']
                extra_dict = dict(alpha=float(refit_from[-9:-4]))
            else:
                extra_dict = {}
            accuracies.extend([dict(seed=seed, study=study, accuracy=score,
                                    **extra_dict) for study, score in
                               test_scores.items()])
            metrics.extend([dict(study=study, contrast=contrast,
                                 prec=precs[study][contrast],
                                 recall=recalls[study][contrast],
                                 f1=f1s[study][contrast],
                                 bacc=baccs[study][contrast],
                                 seed=seed,
                                 **extra_dict
                                 )
                            for study in f1s
                            for contrast in f1s[study]
                            ])
    metrics = pd.DataFrame(metrics)
    metrics = metrics.set_index([*extra_indices, 'study', 'contrast', 'seed'])
    metrics_mean = metrics.groupby(
        [*extra_indices, 'study', 'contrast']).aggregate(
        ['mean', 'std'])
    metrics.to_pickle(join(output_dir, 'metrics.pkl'))
    metrics_mean.to_pickle(join(output_dir, 'metrics_mean.pkl'))

    accuracies = pd.DataFrame(accuracies)
    accuracies = accuracies.set_index([*extra_indices, 'study', 'seed'])[
        'accuracy']
    accuracies_mean = accuracies.groupby([*extra_indices, 'study']).aggregate(
        ['mean', 'std'])
    print(metrics_mean)
    accuracies.to_pickle(join(output_dir, 'accuracies.pkl'))
    accuracies_mean.to_pickle(join(output_dir, 'accuracies_mean.pkl'))


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
        n_subjects[study] = int(ceil(len(this_target['subject'].unique()) / 2))

    chance_level = pd.Series(chance_level)
    n_subjects = pd.Series(n_subjects)
    return chance_level, n_subjects


def get_studies():
    source_dir = join(get_data_dir(), 'reduced_512')
    _, target = load_data_from_dir(data_dir=source_dir)
    studies = list(target.keys())
    return studies


def compare_accuracies():
    metrics = pd.read_pickle(join(get_output_dir(),
                                  'factored_refit_gm_notune',
                                  'metrics.pkl'))
    ref_metrics = pd.read_pickle(join(get_output_dir(),
                                      'logistic_gm', 'metrics.pkl'))
    metrics = metrics.loc[0.0001]
    joined = pd.concat([metrics, ref_metrics], axis=1,
                       keys=['factored', 'baseline'], join='inner')
    diff = joined['factored'] - joined['baseline']
    for v in diff.columns:
        joined['diff', v] = diff[v]
    joined = joined.reset_index()
    joined.to_pickle(join(get_output_dir(), 'joined_contrast.pkl'))


if __name__ == '__main__':
    # gather_factored(join(get_output_dir(), 'factored_gm'))
    gather_factored(join(get_output_dir(), 'logistic_gm'), flavor='single_study')
    gather_factored(join(get_output_dir(), 'factored_refit_gm_notune'),
                    flavor='refit')
    compare_accuracies()
    # gather_factored(join(get_output_dir(), 'factored_refit_gm'), flavor='refit')
    # gather_factored(join(get_output_dir(), 'factored_refit_gm_low_lr'), flavor='refit')
