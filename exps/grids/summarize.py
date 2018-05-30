# Baseline logistic
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from matplotlib import gridspec
from os.path import expanduser, join

idx = pd.IndexSlice


def summarize_baseline():
    output_dir = expanduser('~/output/cogspaces/baseline_logistic_avg')
    #
    # regex = re.compile(r'[0-9]+$')
    # res = []
    # i = 0
    # for this_dir in filter(regex.match, os.listdir(output_dir)):
    #     this_exp_dir = join(output_dir, this_dir)
    #     this_dir = int(this_dir)
    #     try:
    #         config = json.load(
    #             open(join(this_exp_dir, 'config.json'), 'r'))
    #         run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
    #     except (FileNotFoundError, json.decoder.JSONDecodeError):
    #         print('Skipping exp %i' % this_dir)
    #         continue
    #     study = config['data']['studies']
    #     l2_penalty = config['logistic']['l2_penalty']
    #     seed = config['seed']
    #     if run['result'] is None:
    #         continue
    #     else:
    #         test_score = run['result'][study]
    #     res.append(dict(study=study, test_score=test_score,
    #                     seed=seed, l2_penalty=l2_penalty,
    #                     run=this_dir))
    #     print('File %i' % i)
    #     i += 1
    # res = pd.DataFrame(res)
    # pd.to_pickle(res, join(expanduser('~/output/cogspaces/'
    #                                   'baseline.pkl')))
    res = pd.read_pickle(join(expanduser('~/output/cogspaces/baseline.pkl')))
    res = res.set_index(['study', 'seed', 'l2_penalty'])
    idxmax = \
    res.groupby(level=['study', 'l2_penalty']).aggregate('mean').groupby(
        level='study').aggregate('idxmax')['test_score']
    res['l2_penalty_'] = res.index.get_level_values(level='l2_penalty')
    idxmax = idxmax.values.tolist()
    best_res = []
    for study, l2_penalty in idxmax:
        best_res.append(res.loc[idx[study, :, l2_penalty]])
    best = pd.concat(best_res, keys=[study for study, _ in idxmax],
                     names=['study'])[['test_score', 'l2_penalty_']]
    pd.to_pickle(best, join(expanduser('~/output/cogspaces/'
                                      'baseline_seed.pkl')))
    res = best.groupby('study').aggregate(['mean', 'std'])
    pd.to_pickle(res, join(expanduser('~/output/cogspaces/'
                                      'baseline_avg.pkl')))


def summarize_variational():
    output_dir = [expanduser('~/output/cogspaces/variational_3'), ]

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
            test_scores = run['result']
            if test_scores is None:
                print('Skipping exp %i' % this_dir)
                continue
            seed = config['seed']
            this_res = dict(seed=seed, **test_scores)
            res.append(this_res)
    res = pd.DataFrame(res)
    res = res.set_index('seed')
    pd.to_pickle(res, join(expanduser('~/output/cogspaces/variational_3.pkl')))
    res = pd.read_pickle(
        join(expanduser('~/output/cogspaces/variational_3.pkl')))
    studies = res.columns
    res = [res[study] for study in studies]
    res = pd.concat(res, keys=studies, names=['study'])
    pd.to_pickle(res,
                 join(expanduser('~/output/cogspaces/variational_seed_3.pkl')))
    res = res.groupby('study').aggregate(['mean', 'std'])
    pd.to_pickle(res,
                 join(expanduser('~/output/cogspaces/variational_avg_3.pkl')))


def compare_variational():
    variational = pd.read_pickle(
        join(expanduser('~/output/cogspaces/variational_seed_3.pkl')))
    baseline = pd.read_pickle(
        join(expanduser('~/output/cogspaces/baseline_seed.pkl')))['test_score']
    print(baseline)
    print(variational)
    variational = pd.DataFrame(data=dict(score=variational))
    baseline = pd.DataFrame(data=dict(score=baseline))
    joined = variational.join(baseline, how='inner', lsuffix='_variational',
                     rsuffix='_baseline')
    joined['diff'] = joined['score_variational'] - joined['score_baseline']
    joined = joined.groupby('study').aggregate(['mean', 'std'])
    pd.to_pickle(joined, (expanduser('~/output/cogspaces/joined_3.pkl')))


def plot_variational():
    output_dir = expanduser('~/output/cogspaces/')
    data = pd.read_pickle(join(output_dir, 'joined_3.pkl'))
    data = data.sort_values(('diff', 'mean'), ascending=False)
    gs = gridspec.GridSpec(2, 1,
                           height_ratios=[1, 2]
                           )
    gs.update(top=0.98, bottom=0.3, left=0.1, right=0.98)
    fig = plt.figure(figsize=(10, 7))
    ax2 = fig.add_subplot(gs[1])
    ax1 = fig.add_subplot(gs[0], sharex=ax2)
    n_study = data.shape[0]

    ind = np.arange(n_study) * 2 + .5
    width = 1.2
    diff_color = plt.get_cmap('tab10').colors[2]
    baseline_color = plt.get_cmap('tab10').colors[0]
    transfer_color = plt.get_cmap('tab10').colors[1]
    rects = ax1.bar(ind, data[('diff', 'mean')], width,
                    color=diff_color, alpha=0.8)
    errorbar = ax1.errorbar(ind, data[('diff', 'mean')],
                            yerr=data[('diff', 'std')], elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=diff_color,
                            alpha=.8)
    ax1.set_ylabel('Transfer gain')
    ax1.spines['bottom'].set_position('zero')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ind = np.arange(n_study) * 2

    width = .8
    rects1 = ax2.bar(ind, data[('score_baseline', 'mean')], width,
                     color=baseline_color, alpha=.8)
    errorbar = ax2.errorbar(ind, data[('score_baseline', 'mean')],
                            yerr=data[('score_baseline', 'std')], elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=baseline_color,
                            alpha=.8)
    rects2 = ax2.bar(ind + width, data[('score_variational', 'mean')], width,
                     color=transfer_color, alpha=.8)
    errorbar = ax2.errorbar(ind + width, data[('score_variational', 'mean')],
                            yerr=data[('score_variational', 'std')], elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=transfer_color,
                            alpha=.8)
    ax2.set_ylabel('Test accuracy')
    ax2.set_xticks(ind + width / 2)
    ax2.set_xticklabels(data.index.values, rotation=60, ha='right',
                        va='top')
    ax2.set_ylim([0.1, 0.94])
    ax1.legend((rects1[0], rects2[0], rects[0]),
               ('Baseline', 'Transfer', 'Diff'))
    sns.despine(fig)
    plt.savefig(join(output_dir, 'comparison_variational_3.pdf'))
    # plt.show()


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


def summarize_study_selection():
    output_dir = [expanduser('~/output/cogspaces/study_selection_5'), ]

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
            this_res = dict(run=this_dir, seed=config['seed'])
            target_study = config['data']['target_study']
            this_res['target_study'] = target_study
            this_res['study_weight'] = config['model']['study_weight']
            studies = config['data']['studies']
            if isinstance(studies, list):
                studies = 'baseline'
            this_res['studies'] = studies
            test_scores = run['result']
            if test_scores is None:
                continue
            this_res['score'] = test_scores[target_study]
            res.append(this_res)
    res = pd.DataFrame(res)
    res.set_index(['studies', 'target_study', 'seed', 'study_weight'],
                  inplace=True)
    res = res.sort_index()
    # print(res.query("target_study == 'archi'"))

    diff = res.loc['all', 'score'] - res.loc['baseline', 'score']
    transfer = res.loc['all', 'score']
    baseline = res.loc['baseline', 'score']
    summary = pd.DataFrame(data=dict(diff=diff, transfer=transfer,
                                     baseline=baseline))
    summary = summary.groupby(['study_weight', 'target_study']).agg(
        ['mean', 'std'])
    print(summary.loc['sqrt_sample'])
    print(summary.loc['study'])
    # print(summary)
    pd.to_pickle(summary, join(expanduser('~/output/cogspaces'
                                          '/study_selection_5.pkl')))


def plot_study_selection():
    output_dir = expanduser('~/output/cogspaces/')
    data = pd.read_pickle(join(output_dir, 'study_selection.pkl'))
    data = data.loc['sqrt_sample']
    data = data.sort_values(('diff', 'mean'), ascending=False)
    gs = gridspec.GridSpec(2, 1,
                           height_ratios=[1, 2]
                           )
    gs.update(top=0.98, bottom=0.3, left=0.1, right=0.98)
    fig = plt.figure(figsize=(10, 7))
    ax2 = fig.add_subplot(gs[1])
    ax1 = fig.add_subplot(gs[0], sharex=ax2)
    n_study = data.shape[0]

    ind = np.arange(n_study) * 2 + .5
    width = 1.2
    diff_color = plt.get_cmap('tab10').colors[2]
    baseline_color = plt.get_cmap('tab10').colors[0]
    transfer_color = plt.get_cmap('tab10').colors[1]
    rects = ax1.bar(ind, data[('diff', 'mean')], width,
                    color=diff_color, alpha=0.8)
    errorbar = ax1.errorbar(ind, data[('diff', 'mean')],
                            yerr=data[('diff', 'std')], elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=diff_color,
                            alpha=.8)
    ax1.set_ylabel('Transfer gain')
    ax1.spines['bottom'].set_position('zero')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ind = np.arange(n_study) * 2

    width = .8
    rects1 = ax2.bar(ind, data[('baseline', 'mean')], width,
                     color=baseline_color, alpha=.8)
    errorbar = ax2.errorbar(ind, data[('baseline', 'mean')],
                            yerr=data[('baseline', 'std')], elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=baseline_color,
                            alpha=.8)
    rects2 = ax2.bar(ind + width, data[('transfer', 'mean')], width,
                     color=transfer_color, alpha=.8)
    errorbar = ax2.errorbar(ind + width, data[('transfer', 'mean')],
                            yerr=data[('transfer', 'std')], elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=transfer_color,
                            alpha=.8)
    ax2.set_ylabel('Test accuracy')
    ax2.set_xticks(ind + width / 2)
    ax2.set_xticklabels(data.index.values, rotation=60, ha='right',
                        va='top')
    ax2.set_ylim([0.1, 0.94])
    ax1.legend((rects1[0], rects2[0], rects[0]),
               ('Baseline', 'Transfer', 'Diff'))
    sns.despine(fig)
    plt.savefig(join(output_dir, 'comparison_sqrt_sample.pdf'))
    # plt.show()


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
    summarize_variational()
    # summarize_baseline()
    compare_variational()
    plot_variational()
    # summarize_factored    ()
    # summarize_study_selection()
    # plot_study_selection()
    # plot()
