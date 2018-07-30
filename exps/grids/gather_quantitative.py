# Baseline logistic
import json
import os
import re
from math import ceil
from os.path import join

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir

idx = pd.IndexSlice


def gather_factored(output_dir, flavor='simple'):
    regex = re.compile(r'[0-9]+$')
    if flavor == 'refit':
        extra_indices = ['alpha']
    elif flavor == 'reset':
        extra_indices = ['reset']
    elif flavor == 'l2':
        extra_indices = ['l2']
    elif flavor == 'weight_power':
        extra_indices = ['weight_power']
    elif flavor == 'training_curves':
        extra_indices = ['train_size']
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
            Cs, precs, f1s, recalls = load(
                join(this_exp_dir, 'scores.pkl'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue

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

        if flavor in ['single_study', 'transfer', 'training_curves']:
            if flavor == 'single_study':
                study = config['data']['studies']
                extra_dict = {}
            elif flavor == 'transfer':
                study = config['model']['target_study']
                extra_dict = {}
            else:
                train_size = config['data']['train_size']
                for study, this_size in train_size.items():
                    if this_size != 1:
                        extra_dict = {'train_size': this_size}
                        break
            accuracies.append(dict(study=study, accuracy=test_scores[study],
                                   seed=seed, **extra_dict))

            f1s = f1s[study]
            recalls = recalls[study]
            precs = precs[study]
            baccs = baccs[study]

            metrics.extend([dict(study=study, contrast=contrast,
                                 prec=precs[contrast],
                                 recall=recalls[contrast],
                                 bacc=baccs[contrast],
                                 f1=f1s[contrast],
                                 seed=seed,
                                 **extra_dict)
                            for contrast in f1s])
        else:
            if flavor == 'refit':
                refit_from = config['factored']['refit_from']
                extra_dict = dict(alpha=float(refit_from[-9:-4]))
            elif flavor == 'reset':
                reset = config['factored']['reset_classifiers']
                extra_dict = dict(reset=reset)
            elif flavor == 'l2':
                l2 = config['factored']['l2_penalty']
                extra_dict = dict(l2=l2)
            elif flavor == 'weight_power':
                weight_power = config['factored']['weight_power']
                extra_dict = dict(weight_power=weight_power)
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
    print(accuracies)
    accuracies = accuracies.set_index([*extra_indices, 'study', 'seed'])[
        'accuracy']
    accuracies_mean = accuracies.groupby([*extra_indices, 'study']).aggregate(
        ['mean', 'std'])
    accuracies.to_pickle(join(output_dir, 'accuracies.pkl'))
    print(accuracies_mean)
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


def get_full_subjects():
    _, target = load_data_from_dir(
        data_dir=join(get_data_dir(), 'reduced_512'))
    n_subjects = {}
    for study, this_target in target.items():
        n_subjects[study] = len(this_target['subject'].unique())
    return n_subjects


def get_studies():
    source_dir = join(get_data_dir(), 'reduced_512')
    _, target = load_data_from_dir(data_dir=source_dir)
    studies = list(target.keys())
    return studies


if __name__ == '__main__':
    launch = [
        # delayed(gather_factored)(join(get_output_dir(), 'factored')),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_l2_no_rank'), flavor='l2'),
        # delayed(gather_factored)(join(get_output_dir(), 'training_curves'), flavor='training_curves'),
        # delayed(gather_factored)(join(get_output_dir(), 'logistic_tc'), flavor='training_curves'),
        delayed(gather_factored)(join(get_output_dir(), 'logistic_tc_2'), flavor='training_curves'),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_l2'), flavor='l2'),
        # delayed(gather_factored)(join(get_output_dir(), 'weight_power'), flavor='weight_power'),
        # delayed(gather_factored)(join(get_output_dir(), 'adaptive_dropout')),
        # delayed(gather_factored)(join(get_output_dir(), 'bn')),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_refit_gm_normal_init_low_lr'), flavor='refit'),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_refit_gm_normal_init_positive_notune'), flavor='refit'),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_refit_gm_normal_init_rest_positive_notune'), flavor='refit'),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_refit_gm_normal_init_positive_notune'), flavor='refit'),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_refit_gm_rest_positive_notune'), flavor='refit'),
        # delayed(gather_factored)(join(get_output_dir(), 'full_logistic'), flavor='single_study'),
        # delayed(gather_factored)(join(get_output_dir(), 'logistic_gm'), flavor='single_study'),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_gm_single'), flavor='single_study'),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_refit_gm_notune'), flavor='refit'),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_refit_gm_normal_init_notune'), flavor='refit'),
        # delayed(gather_factored)(join(get_output_dir(), 'factored_refit_gm_low_lr'), flavor='refit')
        ]
    Parallel(n_jobs=7)(launch)
