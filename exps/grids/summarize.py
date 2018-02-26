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

    pd.to_pickle(max_res, join(expanduser('~/output/cogspaces/'
                                          'baseline_logistic.pkl')))


def summarize_mtl():
    output_dir = [expanduser('~/output/cogspaces/multi_studies_paradox'),
                  expanduser('~/output/cogspaces/multi_studies_paradigm')]

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

    res = res.query("optimizer == 'adam' and dropout == 0.80")
    print(res)
    pd.to_pickle(res, join(expanduser('~/output/cogspaces/'
                                      'mtl.pkl')))


if __name__ == '__main__':
    summarize_mtl()
