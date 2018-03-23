# Baseline logistic
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
from joblib import delayed, Parallel
from os.path import expanduser, join
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_output_dir, get_data_dir

idx = pd.IndexSlice


def get_studies_list(exp='all_pairs_4'):
    res = pd.read_pickle(join(expanduser('~/output/cogspaces/%s.pkl' % exp)))

    baseline = res.reset_index()
    baseline = baseline.loc[baseline['target'] == baseline['help']]
    baseline.set_index(['target', 'seed'], inplace=True)
    baseline.drop(columns=['help'], inplace=True)
    baseline.sort_index(inplace=True)

    result = pd.merge(res.reset_index(), baseline.reset_index(),
                      suffixes=('', '_baseline'),
                      on=['target', 'seed'], how='outer').set_index(
        ['target', 'help', 'seed'])
    result.sort_index(inplace=True)
    result['diff'] = result['score'] - result['score_baseline']
    diff = result['diff'].groupby(level=['target', 'help']).agg(
        {'mean': np.mean, 'std': np.std})
    studies_list = diff.index.get_level_values('target').unique().values

    def trim_neg_and_norm(x):
        x = x.loc[x > 0].reset_index('target', drop=True)
        return x

    positive = diff['mean'].groupby('target').apply(trim_neg_and_norm)

    res = []
    for target in studies_list:
        try:
            help = positive.loc[target]
        except:
            help = []
        studies = [target] + help.index.get_level_values('help').values.tolist()
        res.append(studies)
    return res


def summarize_all_positive(exp='all_pairs_positive_transfer'):
    output_dir = [expanduser('~/output/cogspaces/%s' % exp), ]

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
            studies = config['data']['studies']
            seed = config['seed']
            test_scores = run['result']
            if test_scores is None:
                test_scores = info['test_scores'][-1]
                this_res = [dict(target=studies[0],
                                 help=' '.join(studies[1:]),
                                 score=test_scores[studies[0]],
                                 seed=seed)]
                res += this_res
    res = pd.DataFrame(res)
    res.set_index(['target', 'help', 'seed'], inplace=True)
    res = res.sort_index()
    print(res)
    pd.to_pickle(res, join(expanduser('~/output/cogspaces/%s.pkl' % exp)))


def summarize_all_pairs(exp='all_pairs_4'):
    output_dir = [expanduser('~/output/cogspaces/%s' % exp), ]

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
            studies = config['data']['studies']
            seed = config['seed']
            test_scores = run['result']
            if test_scores is None:
                test_scores = info['test_scores'][-1]
            if len(studies) > 1:
                this_res = [dict(target=studies[0],
                                 help=studies[1],
                                 score=test_scores[studies[0]],
                                 seed=seed),
                            dict(target=studies[1], help=studies[0],
                                 score=test_scores[studies[1]],
                                 seed=seed)]
            else:
                this_res = [dict(target=studies[0], help=studies[0],
                                 score=test_scores[studies[0]],
                                 seed=seed)]
            res += this_res
    res = pd.DataFrame(res)
    res.set_index(['target', 'help', 'seed'], inplace=True)
    res = res.sort_index()

    pd.to_pickle(res, join(expanduser('~/output/cogspaces/%s.pkl' % exp)))
    res = pd.read_pickle(join(expanduser('~/output/cogspaces/%s.pkl' % exp)))

    baseline = res.reset_index()
    baseline = baseline.loc[baseline['target'] == baseline['help']]
    baseline.set_index(['target', 'seed'], inplace=True)
    baseline.drop(columns=['help'], inplace=True)
    baseline.sort_index(inplace=True)

    result = pd.merge(res.reset_index(), baseline.reset_index(),
                      suffixes=('', '_baseline'),
                      on=['target', 'seed'], how='outer').set_index(
        ['target', 'help', 'seed'])
    result.sort_index(inplace=True)
    result['diff'] = result['score'] - result['score_baseline']
    diff = result['diff'].groupby(level=['target', 'help']).agg(
        {'mean': np.mean, 'std': np.std})
    print(result.loc['amalric2012mathematicians'])
    diff_mean = diff['mean'].groupby(level='help').aggregate(
        'mean').sort_values(ascending=True)
    studies_list = diff_mean.index.get_level_values('help').unique().values
    n_studies = len(studies_list)

    scores = np.zeros((n_studies, n_studies, 2))
    for i in range(n_studies):
        for j in range(n_studies):
            scores[i, j, :] = diff.loc[(studies_list[i],
                                        studies_list[j])]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    caxes = []
    scores_max = scores[:, :, 0] + .1 * scores[:, :, 1]
    scores_min = scores[:, :, 0] - .1 * scores[:, :, 1]
    scores_mean = scores[:, :, 0]
    vmax = max(np.max(np.abs(scores_max)), np.max(np.abs(scores_min)))
    for ax, these_scores, title in zip(axes,
                                       [scores_min, scores_mean, scores_max],
                                       ['mean - std',
                                        'mean test accuracy gain',
                                        'mean + std']):
        cax = ax.matshow(these_scores, vmax=vmax, vmin=-vmax,
                         cmap=plt.get_cmap('RdBu_r'))
        caxes.append(cax)
        ax.tick_params(axis='x', labelbottom=True, labeltop=False, top=False,
                       bottom=True)
        ax.set_xticks(np.arange(n_studies))
        ax.set_xticklabels(studies_list, rotation=60, va='top', ha='right')
        ax.set_yticks(np.arange(n_studies))
        ax.set_yticklabels([])
        ax.set_xlabel('Helper dataset')
        ax.annotate(title, xy=[.5, 1.03], xycoords='axes fraction',
                    ha='center')
    axes[0].set_yticklabels(studies_list)
    axes[0].set_ylabel('Target dataset')
    fig.colorbar(caxes[2], ax=axes[2])
    plt.subplots_adjust(top=0.95, bottom=0.35, left=0.15, right=0.98,
                        wspace=0.05)

    plt.savefig(join(get_output_dir(), 'transfer_%s.pdf' % exp))


def make_features():
    source_dir = join(get_data_dir(), 'reduced_512_lstsq')

    res = pd.read_pickle(
        join(expanduser('~/output/cogspaces/all_pairs_4.pkl')))

    baseline = res.reset_index()
    baseline = baseline.loc[baseline['target'] == baseline['help']]
    baseline.set_index(['target', 'seed'], inplace=True)
    baseline.drop(columns=['help'], inplace=True)
    baseline.sort_index(inplace=True)

    comp_df = pd.merge(res.reset_index(), baseline.reset_index(),
                       suffixes=('', '_baseline'),
                       on=['target', 'seed'], how='outer').set_index(
        ['target', 'help', 'seed'])
    comp_df.sort_index(inplace=True)
    comp_df['diff'] = comp_df['score'] - comp_df['score_baseline']
    diff = comp_df['diff'].groupby(level=['target', 'help']).agg(
        {'mean': np.mean, 'std': np.std})
    diff = diff.reset_index()

    data, target = load_data_from_dir(data_dir=source_dir)
    studies_list = list(data.keys())
    n_studies = len(studies_list)
    unary_features = []
    for study, these_data in data.items():
        n_samples = len(these_data)
        test_accuracy = baseline.loc[study, 'score'].mean()
        unary_features.append(dict(study=study, n_samples=n_samples,
                                   test_accuracy=test_accuracy))
    unary_features = pd.DataFrame(unary_features)
    features = []
    distances = Parallel(n_jobs=20, verbose=10)(
        delayed(get_distances)(data[studies_list[i]],
                                   data[studies_list[j]])
        for i in range(n_studies) for j in range(i))
    k = 0
    for i in range(n_studies):
        for j in range(i):
            wass, corr, target_scale, help_scale = distances[k]
            features.append(dict(target=studies_list[i], help=studies_list[j],
                                 wass=wass, corr=corr,
                                 target_scale=target_scale,
                                 help_scale=help_scale,
                                 ))
            features.append(dict(target=studies_list[j], help=studies_list[i],
                                 wass=wass, corr=corr,
                                 target_scale=target_scale,
                                 help_scale=help_scale))
            k += 1

    features = pd.DataFrame(features)
    features = pd.merge(features, unary_features,
                        suffixes=['', ''],
                        left_on=['help'], right_on=['study'], how='outer')
    features = features.drop('study', axis=1)
    features = pd.merge(features, unary_features,
                        suffixes=['_help', '_target'],
                        left_on=['target'], right_on=['study'], how='outer')
    features = features.drop('study', axis=1)

    result = pd.merge(diff, features, on=['target', 'help'], how='inner')
    pd.to_pickle(result, join(get_output_dir(), 'features_transfer.pkl'))


def plot_features():
    data = pd.read_pickle(join(get_output_dir(), 'features_transfer.pkl'))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
    mean = np.array(data['mean'].values)
    ax1.scatter(data['n_samples_target'], data['n_samples_help'],
                c=mean, cmap=plt.get_cmap('RdBu_r'), alpha=1)
    ax1.set_xlabel('Target # sample')
    ax1.set_ylabel('Helper # sample')
    ax1.set_xlim([0, 1000])
    ax1.set_ylim([0, 1000])
    ax2.scatter(data['test_accuracy_target'], data['test_accuracy_help'],
                c=mean, cmap=plt.get_cmap('RdBu_r'), alpha=1)
    ax2.set_xlabel('Target test accuracy')
    ax2.set_ylabel('Helper test accuracy')
    ax3.scatter(data['wass'], data['mean'], s=0.5)
    ax3.set_xlim([500, 800])
    ax3.set_ylabel('Transfer')
    ax3.set_xlabel('Dataset distance')
    plt.show()

    X, y = data['wass'][:, None], data['mean']
    lr = LinearRegression().fit(X, y)
    y_pred = lr.predict(X)
    r2 = r2_score(y, y_pred)
    print('Sinkhorn -> Transfer r2', r2)

    X, y = data[['n_samples_help',
                 'n_samples_target',
                 'test_accuracy_target',
                 'test_accuracy_help',
                 'wass',
                 'corr',
                 'help_scale',
                 'target_scale'
                 ]], data['mean']
    lr = LinearRegression().fit(X, y)
    y_pred = lr.predict(X)
    r2 = r2_score(y, y_pred)
    print('All -> Transfer r2', r2)

    data = pd.read_pickle(join(get_output_dir(), 'features_transfer.pkl'))
    transfer_mean = data[['help', 'mean',
                          'n_samples_help',
                          'test_accuracy_help']].groupby(by='help').agg(
        {'mean': 'mean', 'n_samples_help': 'max', 'test_accuracy_help': 'max'})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    ax1.scatter(transfer_mean['n_samples_help'], transfer_mean['mean'])
    ax1.set_xlabel('Helper # sample')
    ax1.set_ylabel('Mean transfer')

    ax2.scatter(transfer_mean['test_accuracy_help'], transfer_mean['mean'])
    ax2.set_xlabel('Helper test accuracy')
    ax2.set_ylabel('Mean transfer')

    plt.show()

    X, y = transfer_mean['n_samples_help'][:, None], transfer_mean['mean']
    lr = LinearRegression().fit(X, y)
    y_pred = lr.predict(X)
    r2 = r2_score(y, y_pred)
    print('N samples help -> Mean Transfer r2', r2)

    X, y = transfer_mean['test_accuracy_help'][:, None], transfer_mean['mean']
    lr = LinearRegression().fit(X, y)
    y_pred = lr.predict(X)
    r2 = r2_score(y, y_pred)
    print('Test accuracy help -> Mean Transfer r2', r2)


def get_distances(target_data, help_data):
    target_sc = StandardScaler()
    help_sc = StandardScaler()
    target_data = target_sc.fit_transform(target_data)
    help_data = help_sc.fit_transform(help_data)

    D = ot.dist(target_data, help_data)
    n, m = len(target_data), len(help_data)
    a, b = np.ones(n) / n, np.ones(m) / m  # uniform distribution on samples
    P = ot.sinkhorn(a, b, D / D.max(), reg=1e-3)
    wass = np.sum(P * D)

    gram = target_data.dot(help_data.T)
    gram = np.abs(gram)
    corr = np.mean(gram)
    return wass, corr, target_sc.scale_, help_sc.scale_


if __name__ == '__main__':
    # summarize_all_pairs('all_pairs_advers')
    # summarize_all_pairs('all_pairs_4')
    # make_features()
    # plot_features()
    summarize_all_positive()