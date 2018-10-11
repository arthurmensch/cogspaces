import os
from os.path import join, expanduser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cogspaces.datasets.derivative import get_chance_subjects
from matplotlib import gridspec, ticker

output_dir = expanduser(join('~', 'output'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

idx = pd.IndexSlice

save_dir = join(output_dir, 'compare')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pad_bottom = .51
pad_top = .02
pad_right = .02
pad_left = 2.49


def make_data():
    factored_output_dir = join(output_dir, 'factored')
    baseline_output_dir = join(output_dir, 'logistic')

    factored = pd.read_pickle(join(factored_output_dir, 'accuracies.pkl'))
    baseline = pd.read_pickle(join(baseline_output_dir, 'accuracies.pkl'))

    chance_level, n_subjects = get_chance_subjects()

    joined = pd.concat([factored, baseline, chance_level, n_subjects],
                       keys=['factored', 'baseline'], axis=1)
    joined['diff'] = joined['factored'] - joined['baseline']
    joined_mean = joined.groupby('study').aggregate(['mean', 'std'])
    joined_mean['chance'] = chance_level
    joined_mean['n_subjects'] = n_subjects

    data = joined_mean
    data = data.sort_values(('diff', 'mean'), ascending=True)
    sort = data.index.values.tolist()[::-1]
    return data, sort


def plot_joined(data):
    width, height = 8.5, 5.6
    df = pd.read_csv('~/work/papers/papers/thesis/brainpedia.csv', index_col=0,
                     header=0)
    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[2.7, 1]
                           )
    fig = plt.figure(figsize=(width, height))
    gs.update(left=2 / width, right=1 - .2 / width,
              bottom=pad_bottom / height, top=1 - pad_top / height,
              wspace=1.2 / width
              )
    ax2 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax2)
    n_study = data.shape[0]

    ind = np.arange(n_study) * 2 + .5
    width = 1.2
    diff_color = sns.color_palette("husl", n_study)[::-1]
    diff_color_err = [(max(0, r - .1), max(0, g - .1), max(0, b - .1))
                      for r, g, b in diff_color]

    baseline_color = '0.75'
    baseline_color_err = '0.65'
    transfer_color = '0.1'
    transfer_color_err = '0.'
    ax1.barh(ind, data[('diff', 'mean')], width,
             color=diff_color)
    for this_x, this_y, this_xerr, this_color in zip(data[('diff', 'mean')],
                                                     ind,
                                                     data[('diff', 'std')],
                                                     diff_color_err):
        ax1.errorbar(this_x, this_y, xerr=this_xerr, elinewidth=1.5,
                     capsize=2, linewidth=0, ecolor=this_color,
                     alpha=.5)
    ax1.set_xlabel('Multi-study acc. gain', fontsize=15)
    ax1.spines['left'].set_position('zero')
    plt.setp(ax1.get_yticklabels(), visible=False)

    ax1.set_xlim([-0.06, 0.19])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    ind = np.arange(n_study) * 2

    width = .8
    rects1 = ax2.barh(ind, data[('baseline', 'mean')], width,
                      color=baseline_color)
    ax2.errorbar(data[('baseline', 'mean')], ind,
                 xerr=data[('baseline', 'std')],
                 elinewidth=1.5,
                 capsize=2, linewidth=0, ecolor=baseline_color_err,
                 alpha=.5)
    rects2 = ax2.barh(ind + width, data[('factored', 'mean')], width,
                      color=transfer_color)
    ax2.errorbar(data[('factored', 'mean')], ind + width,
                 xerr=data[('factored', 'std')],
                 elinewidth=1.5,
                 capsize=2, linewidth=0, ecolor=transfer_color_err,
                 alpha=.5)

    lines = ax2.vlines([data['chance']],
                       ind - width / 2, ind + 3 * width / 2, colors='r',
                       linewidth=1, linestyles='--')

    ax2.set_ylim([-1, 2 * data.shape[0]])
    plt.setp(ax2.yaxis.get_ticklabels(), fontsize=10)
    ax2.set_xlim([0., 0.95])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.set_xlabel('Decoding accuracy on test set', fontsize=15)

    handles = [rects1, rects2, lines]
    labels = ['Decoding from\nvoxels',
              'Decoding from\nmulti-study\nnetworks',
              'Chance level']
    ax2.legend(handles, labels, loc='lower right', frameon=False,
               fontsize=13,
               bbox_to_anchor=(1.16, .64))

    ax2.set_yticks(ind + width / 2)
    ax2.annotate('Task fMRI study', xy=(-.4, -.09), xycoords='axes fraction',
                 fontsize=15)
    labels = [
        '%s' % df.loc[label]['Description'] for label in data.index.values]
    ax2.set_yticklabels(labels, ha='right', va='center', fontsize=8.5)
    sns.despine(fig)

    plt.savefig(join(save_dir, 'joined_mean.pdf'), facecolor=None,
                edgecolor=None,
                transparent=True)
    plt.close(fig)

    return sort


def plot_compare_methods(sort, ablation=None):
    width, height = 6.2, 3

    exps = ['logistic', 'factored']
    baseline = 'logistic'

    dfs = []
    for exp in exps:
        df = pd.read_pickle(join(output_dir, exp, 'accuracies.pkl'))
        dfs.append(df)

    df = pd.concat(dfs, axis=0, keys=exps, names=['method'])

    df_std = []
    df_mean = []
    crossed_methods = []
    methods = df.index.get_level_values('method').unique()
    for i, method in enumerate(methods):
        df_std.append(
            df.loc[method] - df.loc[baseline].groupby('study').transform(
                'median'))
        for method_ref in methods:
            crossed_methods.append(method + '-' + method_ref)
            diff = df.loc[method] - df.loc[method_ref]
            df_mean.append(diff.agg(['mean', 'std', 'median',
                                     lambda x: (x >= 0).mean()]))
    df_std = pd.concat(df_std, keys=methods, names=['method'])
    df_mean = pd.concat(df_mean, keys=crossed_methods, names=['method'])

    sort = pd.MultiIndex.from_product([methods, sort],
                                      names=['method', 'study'])
    df_sort = df_std.reset_index('seed').loc[sort]

    df_sort = df_sort.reset_index()

    n_studies = len(df_sort['study'].unique())

    diff_color = sns.color_palette("husl", n_studies)
    if ablation is not None:
        height = height * len(exps) / 3
        if ablation == 'transfer':
            height *= 1.1

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    if ablation is None:
        fig.subplots_adjust(left=.07 / width, right=1 - 1.9 / width,
                            bottom=0.55 / height, top=1 - pad_top / height, )
    else:
        fig.subplots_adjust(left=.11 / width, right=1 - 1.9 / width,
                            bottom=0.55 / height, top=1 - pad_top / height, )

    params = dict(x="accuracy", y="method", hue="study",
                  data=df_sort, dodge=True, ax=ax,
                  palette=diff_color
                  )
    sns.stripplot(alpha=1, zorder=200, size=3.5, linewidth=0, jitter=True,
                  **params)
    sns.boxplot(x="accuracy", y="method", color='0.75', showfliers=False,
                whis=1.5,
                data=df_sort, ax=ax, zorder=100)
    ax.set_xlim([-0.14, 0.175])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel('Accuracy gain compared to baseline median', fontsize=15)

    ax.set_ylabel('')
    ax.spines['left'].set_position('zero')
    plt.setp(ax.spines['left'], zorder=2)
    ax.set_yticklabels([])
    ax.set_yticks([])

    methods = df_sort['method'].unique()

    y_labels = {}
    y_labels['factored'] = ('Factored decoder')
    y_labels['logistic'] = ('Standard decoding\nfrom voxels')

    for i, method in enumerate(methods):
        if i == len(methods) - 1:
            fontweight = 'bold'
        else:
            fontweight = 'normal'
        ax.annotate(y_labels[method], xy=(1, i), xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=15 if method != 'factored_transfer' else 13,
                    fontweight=fontweight,
                    xycoords=('axes fraction', 'data'),
                    va='center',
                    ha='left')

    sns.despine(fig)

    plt.setp(ax.legend(), visible=False)
    plt.savefig(join(save_dir, 'comparison_method%s.pdf' % (
        '_%s' % ablation if ablation is not None else '')))
    plt.close(fig)


def plot_gain_vs_size(sort):
    width, height = 3, 2.3

    colors = sns.color_palette('husl', len(sort))
    colors = {study: color for study, color in zip(sort, colors)}

    factored_output_dir = join(output_dir, 'factored')
    baseline_output_dir = join(output_dir, 'logistic')

    factored = pd.read_pickle(join(factored_output_dir, 'accuracies.pkl'))
    baseline = pd.read_pickle(join(baseline_output_dir, 'accuracies.pkl'))

    chance_level, n_subjects = get_chance_subjects()
    joined = pd.concat([factored, baseline],
                       keys=['factored', 'baseline', ], axis=1)
    joined['diff'] = joined['factored'] - joined['baseline']
    joined = joined.reset_index('study')
    joined = joined.assign(chance=lambda x:
    list(map(lambda y: chance_level.loc[y], x.study)),
                           n_subjects=lambda x:
                           list(map(lambda y: n_subjects.loc[y], x.study)))
    joined = joined.set_index(['study'])
    joined_mean = joined.groupby('study').aggregate(['mean', 'std'])

    fig, ax = plt.subplots(1, 1, figsize=(width, height), )

    sns.regplot(joined_mean['n_subjects', 'mean'], joined_mean['diff', 'mean'],
                n_boot=10000, ax=ax, color='black', logx=True,
                order=1, truncate=True, lowess=True,
                scatter=False)
    clouds = ax.scatter(joined_mean['n_subjects', 'mean'],
                        joined_mean['diff', 'mean'],
                        c=list(map(lambda x: colors[x],
                                   joined_mean.index.get_level_values(
                                       'study'))),
                        s=8, )
    ax.set_xscale('log')
    ax.set_ylabel('Multi study acc. gain', fontsize=15)
    ax.set_xlabel('Number of train subjects', fontsize=15, zorder=400)
    ax.set_xlim([3, 420])
    ax.set_xticks([4, 16, 32, 100, 300, 400])
    ax.set_xticklabels([4, 16, 32, 100, 400])

    ax.legend([clouds], ['Study'], frameon=True, scatterpoints=3, fontsize=13)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.spines['bottom'].set_position('zero')

    fig.subplots_adjust(left=.7 / width, right=1 - pad_right / width,
                        bottom=0.35 / height, top=1 - .08 / height, )

    sns.despine(fig)
    fig.savefig(join(save_dir, 'gain_vs_size.pdf'))


def plot_gain_vs_accuracy(sort):
    width, height = 3, 2.3

    metrics = pd.read_pickle(join(output_dir, 'factored',
                                  'metrics.pkl'))
    metrics = metrics.loc[0.0001]
    ref_metrics = pd.read_pickle(join(output_dir, 'logistic',
                                      'metrics.pkl'))
    joined = pd.concat([metrics, ref_metrics], axis=1,
                       keys=['factored', 'baseline'], join='inner')
    diff = joined['factored'] - joined['baseline']
    for v in diff.columns:
        joined['diff', v] = diff[v]

    mean = joined.groupby(by=['study', 'contrast']).aggregate(['mean', 'std'])
    mean = mean.reset_index()

    colors = sns.color_palette('husl', len(sort))
    colors = {study: color for study, color in zip(sort, colors)}

    for score in ['bacc']:

        fig, ax = plt.subplots(1, 1, figsize=(width, height), )
        fig.subplots_adjust(left=.7 / width, right=1 - .2 / width,
                            bottom=.5 / height, top=1 - .08 / height, )

        sns.regplot(mean['baseline', score, 'mean'],
                    mean['diff', score, 'mean'],
                    n_boot=10, ax=ax, lowess=True, color='black',
                    scatter=False)

        clouds = ax.scatter(mean['baseline', score, 'mean'],
                            mean['diff', score, 'mean'],
                            s=4,
                            c=list(map(lambda x: colors[x], mean['study'])),
                            marker='o')
        if score == 'bacc':
            ax.set_xlim([.49, 1.01])
            ax.set_ylim([-0.05, 0.1])

            ax.yaxis.set_major_formatter(
                ticker.PercentFormatter(xmax=1, decimals=0))
            ax.xaxis.set_major_formatter(
                ticker.PercentFormatter(xmax=1, decimals=0))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        else:
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.05, 0.15])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
        if score == 'bacc':
            ax.set_xlabel('Baseline balanced accuracy $\\star$',
                          fontsize=15)
            ax.xaxis.set_label_coords(0.46, -0.17)
        else:
            ax.set_xlabel('Baseline F1 score (per contrast)')
        ax.set_ylabel('Multi-study b-acc. gain    ', fontsize=15)
        ax.hlines(0, 0, 1, linestyle=(0, [2, 4]))

        ax.legend([clouds], ['Contrast'], frameon=True, scatterpoints=3,
                  fontsize=13,
                  loc='upper right')

        sns.despine(fig)
        fig.savefig(join(save_dir, 'gain_vs_accuracy_%s.pdf' % score))


data, sort = make_data()
plot_joined(data)
plot_gain_vs_accuracy(sort)
plot_gain_vs_size(sort)
plot_compare_methods(sort)
