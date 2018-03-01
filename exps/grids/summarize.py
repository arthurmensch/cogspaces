# Baseline logistic
import json
import os
import re
from os.path import expanduser, join

import numpy as np
import pandas as pd
from joblib import load

from cogspaces.introspect.maps import coefs_from_model

import matplotlib.pyplot as plt



def summarize_baseline():
    output_dir = expanduser('~/output/cogspaces/baseline_logistic_icbm_gm')

    regex = re.compile(r'[0-9]+$')
    res = []
    estimators = []
    for this_dir in filter(regex.match, os.listdir(output_dir)):
        this_exp_dir = join(output_dir, this_dir)
        this_dir = int(this_dir)
        try:
            config = json.load(
                open(join(this_exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
            info = json.load(open(join(this_exp_dir, 'info.json'), 'r'))
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            print('Skipping exp %i' % this_dir)
            continue
        study = config['data']['studies']
        l2_penalty = config['logistic']['l2_penalty']
        if run['result'] is None:
            continue
        else:
            test_score = run['result'][study]
        res.append(dict(study=study, test_score=test_score,
                        run=this_dir))
    res = pd.DataFrame(res)

    max_res = res.groupby(by='study').aggregate('idxmax')['test_score']
    max_res = res.iloc[max_res.values.tolist()]

    coefs = {}
    print(max_res)
    for this_dir in max_res['run']:
        exp_dir = join(output_dir, str(this_dir))
        estimator = load(join(exp_dir, 'estimator.pkl'))
        standard_scaler = load(join(exp_dir, 'standard_scaler.pkl'))
        target_encoder = load(join(exp_dir, 'target_encoder.pkl'))
        dict_coefs, names = coefs_from_model(estimator, target_encoder,
                                              standard_scaler)
        for study, these_coefs in dict_coefs.items():
            # these_coefs -= np.mean(these_coefs, axis=0)[None, :]
            these_coefs /= np.sqrt(np.sum(these_coefs ** 2, axis=1))[:, None]
            coefs[study] = these_coefs
    lengths = np.array([0] + [coef.shape[0] for coef in coefs.values()])
    limits = np.cumsum(lengths)
    ticks = (limits[:-1] + limits[1:]) / 2
    names = max_res['study'].values
    coefs = np.concatenate(list(coefs.values()), axis=0)
    corr = coefs.dot(coefs.T)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.matshow(corr)
    ax.hlines(limits, xmin=0, xmax=limits[-1])
    ax.vlines(limits, ymin=0, ymax=limits[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(names)
    plt.savefig(expanduser('~/output/cogspaces/corr.png'))
    plt.close(fig)
    pd.to_pickle(max_res, join(expanduser('~/output/cogspaces/'
                                          'baseline_logistic_icbm_gm.pkl')))


def summarize_factored():
    output_dir = [expanduser('~/output/cogspaces/factored'), ]

    regex = re.compile(r'[0-9]+$')
    res = []
    for this_output_dir in output_dir:
        for this_dir in filter(regex.match, os.listdir(this_output_dir)):
            this_exp_dir = join(this_output_dir, this_dir)
            this_dir = int(this_dir)
            if 11 <= this_dir <= 16:
                try:
                    config = json.load(
                        open(join(this_exp_dir, 'config.json'), 'r'))
                    run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
                    info = json.load(
                        open(join(this_exp_dir, 'info.json'), 'r'))
                except (FileNotFoundError, json.decoder.JSONDecodeError):
                    print('Skipping exp %i' % this_dir)
                    continue
                estimator = config['model']['estimator']
                studies = config['data']['studies']
                test_scores = run['result']
                this_res = dict(estimator=estimator,
                                run=this_dir)
                this_res['study_weight'] = config['model']['study_weight']
                if estimator == 'factored':
                    this_res['optimizer'] = config['factored']['optimizer']
                    this_res['shared_embedding_size'] = config['factored'][
                        'shared_embedding_size']
                    this_res['private_embedding_size'] = config['factored'][
                        'private_embedding_size']
                    this_res['shared_embedding'] = config['factored'][
                        'shared_embedding']
                    this_res['dropout'] = config['factored']['dropout']
                    this_res['input_dropout'] = config['factored'][
                        'input_dropout']
                else:
                    this_res['optimizer'] = 'fista'
                if studies == 'all' and test_scores is not None:
                    mean_test = np.mean(np.array(
                        list(test_scores.values())))
                    this_res['mean_test'] = mean_test
                    this_res = dict(**this_res, **test_scores)
                    res.append(this_res)
    res = pd.DataFrame(res)
    res.set_index(['optimizer', 'shared_embedding_size',
                   'private_embedding_size', 'shared_embedding',
                   'dropout', 'input_dropout'], inplace=True)
    print(res)
    pd.to_pickle(res, join(expanduser('~/output/cogspaces/'
                                      'factored.pkl')))


def summarize_mtl():
    output_dir = [expanduser('~/output/cogspaces/factored_dropout'), ]

    regex = re.compile(r'[0-9]+$')
    res = []
    for this_output_dir in output_dir:
        for this_dir in filter(regex.match, os.listdir(this_output_dir)):
            this_exp_dir = join(this_output_dir, this_dir)
            this_dir = int(this_dir)
            try:
                config = json.load(
                    open(join(this_exp_dir, 'config.json'), 'r'))
                run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
                info = json.load(open(join(this_exp_dir, 'info.json'), 'r'))
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                print('Skipping exp %i' % this_dir)
                continue
            estimator = config['model']['estimator']
            studies = config['data']['studies']
            test_scores = run['result']
            this_res = dict(estimator=estimator,
                            run=this_dir)
            this_res['study_weight'] = config['data']['study_weight']
            if estimator == 'factored':
                this_res['optimizer'] = config['factored']['optimizer']
                this_res['embedding_dim'] = config['factored'][
                    'embedding_size']
                this_res['dropout'] = config['factored']['dropout']
                this_res['input_dropout'] = config['factored']['input_dropout']
            else:
                this_res['optimizer'] = 'fista'
            if studies == 'all' and test_scores is not None:
                mean_test = np.mean(np.array(
                    list(test_scores.values())))
                this_res['mean_test'] = mean_test
                this_res = dict(**this_res, **test_scores)
                res.append(this_res)
    res = pd.DataFrame(res)
    res.set_index([''])
    pd.to_pickle(res, join(expanduser('~/output/cogspaces/'
                                      'factored_dropout.pkl')))


if __name__ == '__main__':
    # summarize_mtl
    summarize_baseline()
    # summarize_factored()
