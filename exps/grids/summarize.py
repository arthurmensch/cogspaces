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
    res.to_pickle(join(output_dir, 'gathered_seed_split_init.pkl'))


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
    res.to_pickle(join(output_dir, 'gathered_reduced_logistic.pkl'))
    
    
def join_baseline_factored(baseline_output_dir, factored_output_dir):
    factored = pd.read_pickle(join(factored_output_dir,
                                   'gathered_seed_split_init.pkl'))
    factored = factored.groupby(['study', 'seed']).aggregate('mean')
    factored.sort_index(inplace=True)

    baseline = pd.read_pickle(join(baseline_output_dir,
                                   'gathered_reduced_logistic.pkl'))
    baseline = baseline['score']
    indices = baseline.groupby(['study', 'l2_penalty']).aggregate(
        'mean').groupby('study').aggregate('idxmax')
    baseline_ = []
    studies = []
    for study, l2_penalty in indices:
        studies.append(study)
        baseline_.append(baseline.loc[idx[study, l2_penalty, :]])
    baseline = pd.concat(baseline_, keys=studies, names=['study'], axis=0)
    baseline.sort_index(inplace=True)

    _, target = load_data_from_dir(
        data_dir=join(get_data_dir(), 'reduced_512'))
    chance_level = {}
    n_subjects = {}
    for study, this_target in target.items():
        chance_level[study] = 1. / len(this_target['contrast'].unique())
        n_subjects[study] = len(this_target['subject'].unique())

    chance_level = pd.Series(chance_level)
    n_subjects = pd.Series(n_subjects)

    joined = pd.concat([factored, baseline, chance_level, n_subjects],
                       keys=['factored', 'baseline'], axis=1)
    joined['diff'] = joined['factored'] - joined['baseline']
    joined_mean = joined.groupby('study').aggregate(['mean', 'std'])
    joined_mean['chance'] = chance_level
    joined_mean['n_subjects'] = n_subjects

    _, target = load_data_from_dir(
        data_dir=join(get_data_dir(), 'reduced_512'))
    joined.to_pickle(join(get_output_dir(), 'joined.pkl'))
    joined_mean.to_pickle(join(get_output_dir(), 'joined_mean.pkl'))


if __name__ == '__main__':
    gather_seed_split_init(join(get_output_dir(), 'seed_split_init'))
    gather_reduced_logistic(join(get_output_dir(), 'reduced_logistic'))
    join_baseline_factored(join(get_output_dir(), 'reduced_logistic'),
                           join(get_output_dir(), 'seed_split_init'))