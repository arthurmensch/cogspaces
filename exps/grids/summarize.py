# Baseline logistic
import json
import os
import re
from os.path import expanduser, join

import numpy as np
import pandas as pd


def summarize_baseline():
    output_dir = [expanduser('~/output/cogspaces/baseline_logistic')]

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
            study = config['data']['studies']
            l2_penalty = config['logistic']['l2_penalty']
            test_score = run['result'][study]
            res.append(dict(study=study, test_score=test_score,
                            run=this_dir))
    res = pd.DataFrame(res)

    max_res = res.groupby(by='study').aggregate('max')['test_score']
    print(max_res)
    pd.to_pickle(max_res, join(expanduser('~/output/cogspaces/'
                                          'baseline_logistic.pkl')))


def summarize_factored():
    output_dir = [expanduser('~/output/cogspaces/factored'),]


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
                    info = json.load(open(join(this_exp_dir, 'info.json'), 'r'))
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
                    this_res['shared_embedding_size'] = config['factored']['shared_embedding_size']
                    this_res['private_embedding_size'] = config['factored']['private_embedding_size']
                    this_res['shared_embedding'] = config['factored']['shared_embedding']
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
    res.set_index(['optimizer', 'shared_embedding_size',
                   'private_embedding_size', 'shared_embedding',
                   'dropout', 'input_dropout'], inplace=True)
    print(res)
    pd.to_pickle(res, join(expanduser('~/output/cogspaces/'
                                      'factored.pkl')))



def summarize_mtl():
    output_dir = [expanduser('~/output/cogspaces/factored_dropout'),]

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
                this_res['embedding_dim'] = config['factored']['embedding_size']
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
    # summarize_baseline()
    summarize_factored()