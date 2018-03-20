# Baseline logistic
import json
import os
import re
from os.path import expanduser, join

import numpy as np
import pandas as pd
from joblib import load

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.introspect.maps import coefs_from_model

import matplotlib.pyplot as plt
import seaborn as sns


def summarize_baseline():
    output_dir = expanduser('~/output/cogspaces/baseline_logistic')

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
    print(max_res)
    pd.to_pickle(max_res, join(expanduser('~/output/cogspaces/'
                                          'max_baseline.pkl')))
    #
    #
    # coefs = {}
    # # print(max_res)
    # for this_dir in max_res['run']:
    #     exp_dir = join(output_dir, str(this_dir))
    #     estimator = load(join(exp_dir, 'estimator.pkl'))
    #     standard_scaler = load(join(exp_dir, 'standard_scaler.pkl'))
    #     target_encoder = load(join(exp_dir, 'target_encoder.pkl'))
    #     dict_coefs, names = coefs_from_model(estimator, target_encoder,
    #                                           standard_scaler)
    #     for study, these_coefs in dict_coefs.items():
    #         # these_coefs -= np.mean(these_coefs, axis=0)[None, :]
    #         these_coefs /= np.sqrt(np.sum(these_coefs ** 2, axis=1))[:, None]
    #         coefs[study] = these_coefs
    # lengths = np.array([0] + [coef.shape[0] for coef in coefs.values()])
    # limits = np.cumsum(lengths)
    # ticks = (limits[:-1] + limits[1:]) / 2
    # names = max_res['study'].values
    # coefs = np.concatenate(list(coefs.values()), axis=0)
    # corr = coefs.dot(coefs.T)
    # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    # ax.matshow(corr)
    # ax.hlines(limits, xmin=0, xmax=limits[-1])
    # ax.vlines(limits, ymin=0, ymax=limits[-1])
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(names, rotation=90)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(names)
    # plt.savefig(expanduser('~/output/cogspaces/corr.png'))
    # plt.close(fig)


def summarize_all_pairs():
    # output_dir = [expanduser('~/output/cogspaces/all_pairs_2'), ]
    #
    # regex = re.compile(r'[0-9]+$')
    # res = []
    # for this_output_dir in output_dir:
    #     for this_dir in filter(regex.match, os.listdir(this_output_dir)):
    #         this_exp_dir = join(this_output_dir, this_dir)
    #         this_dir = int(this_dir)
    #         try:
    #             config = json.load(
    #                 open(join(this_exp_dir, 'config.json'), 'r'))
    #             run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
    #             info = json.load(
    #                 open(join(this_exp_dir, 'info.json'), 'r'))
    #         except (FileNotFoundError, json.decoder.JSONDecodeError):
    #             print('Skipping exp %i' % this_dir)
    #             continue
    #         studies = config['data']['studies']
    #         seed = config['seed']
    #         test_scores = run['result']
    #         if test_scores is None:
    #             test_scores = info['test_scores'][-1]
    #         if len(studies) > 1:
    #             this_res = [dict(target=studies[0],
    #                              help=studies[1],
    #                              score=test_scores[studies[0]],
    #                              seed=seed),
    #                         dict(target=studies[1], help=studies[0],
    #                              score=test_scores[studies[1]],
    #                              seed=seed)]
    #         else:
    #             this_res = [dict(target=studies[0], help=studies[0],
    #                              score=test_scores[studies[0]],
    #                              seed=seed)]
    #         res += this_res
    # res = pd.DataFrame(res)
    # res.set_index(['target', 'help', 'seed'], inplace=True)
    # res = res.sort_index()
    # pd.to_pickle(res, join(expanduser('~/output/cogspaces/all_pairs.pkl')))
    res = pd.read_pickle(join(expanduser('~/output/cogspaces/all_pairs.pkl')))
    print(res.loc['archi'])
    seeds = res.index.get_level_values('seed').unique().values
    n_seeds = len(seeds)
    source_dir = join(get_data_dir(), 'reduced_512')
    data, target = load_data_from_dir(data_dir=source_dir)
    studies_list = list(data.keys())
    n_studies = len(studies_list)
    scores = np.zeros((n_studies, n_studies, n_seeds))
    df = res.groupby(level=['target', 'help']).aggregate('mean')
    baseline_df = df.loc[[(study, study) for study in studies_list]]
    baseline_df.sort_index(inplace=True)
    df = df.groupby(level='target').aggregate('max')
    print(df)
    print(baseline_df)
    for i in range(n_studies):
        for j in range(n_studies):
            scores[i, j, :] = res.loc[(studies_list[i],
                                       studies_list[j]), 'score']
    single_scores = np.diagonal(scores, axis1=0, axis2=1)
    single_scores = single_scores.T
    scores -= single_scores[:, None, :]
    std_scores = np.std(scores, axis=2)
    mean_scores = np.mean(scores, axis=2)

    std_scores[std_scores == 0] = 1
    z_scores = mean_scores / std_scores

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    for ax, these_scores in zip([ax1, ax2], [mean_scores, z_scores]):
        vmax = these_scores.max()
        print(vmax)
        cax = ax.matshow(these_scores, vmax=vmax, vmin=-vmax,
                         cmap=plt.get_cmap('RdBu_r'))
        ax.tick_params(axis='x', labelbottom=True, labeltop=False, top=False,
                       bottom=True)
        ax.set_xticks(np.arange(n_studies))
        ax.set_xticklabels(studies_list, rotation=90)
        ax.set_yticks(np.arange(n_studies))
        ax.set_yticklabels(studies_list)
        fig.colorbar(cax, ax=ax)
    plt.subplots_adjust(top=0.98, bottom=0.32, left=0.07, right=0.98)

    plt.savefig(join(get_output_dir(), 'transfer.pdf'))


def summarize_factored():
    output_dir = [expanduser('~/output/cogspaces/factored_5'), ]

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
                info = json.load(
                    open(join(this_exp_dir, 'info.json'), 'r'))
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                print('Skipping exp %i' % this_dir)
                continue
            estimator = config['model']['estimator']
            studies = config['data']['studies']
            test_scores = run['result']
            if test_scores is None:
                test_scores = info['test_scores'][-1]
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
                this_res['lr'] = config['factored'][
                    'lr']
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
                   'dropout', 'input_dropout', 'lr', 'estimator', 'run',
                   'study_weight'], inplace=True)
    res = res.sort_index()
    # res = res.query("shared_embedding_size == 256 and shared_embedding == 'hard'")
    pd.to_pickle(res, join(expanduser('~/output/cogspaces/factored.pkl')))
    max = res.apply('max')
    print(max)
    # print(res['mean_test'])
    print(res['mean_test'])
    pd.to_pickle(max, join(expanduser('~/output/cogspaces/max_factored.pkl')))


def plot():
    output_dir = expanduser('~/output/cogspaces/')
    baseline = pd.read_pickle(join(output_dir, 'max_baseline.pkl'))
    baseline = baseline.drop('run', axis=1)
    baseline = baseline.set_index('study')
    factored = pd.read_pickle(join(output_dir, 'max_factored.pkl'))
    factored.name = 'factored'
    res = baseline.join(factored)
    res = res.rename({'test_score': 'baseline'},
                     axis='columns')
    res['diff'] = res['factored'] - res['baseline']
    res = res.sort_values('diff', ascending=False)
    print(res)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    n_study = res.shape[0]
    ind = np.arange(n_study) * 2
    width = .8
    rects1 = ax.bar(ind, res['baseline'], width)
    rects2 = ax.bar(ind + width, res['factored'], width)
    ax.set_ylabel('Test accuracy')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(res.index.values, rotation=60, ha='right',
                       va='top')
    ax.set_ylim([0.16, 0.92])

    ax.legend((rects1[0], rects2[0]), ('Baseline', 'Factored'))
    plt.subplots_adjust(top=0.98, bottom=0.32, left=0.07, right=0.98)
    sns.despine(fig)
    plt.savefig(join(output_dir, 'comparison.pdf'))
    plt.show()


if __name__ == '__main__':
    # summarize_mtl()
    # summarize_baseline()
    summarize_all_pairs()
    # summarize_factored()
    # plot()
