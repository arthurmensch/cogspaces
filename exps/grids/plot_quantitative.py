import json
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.family'] = 'cmss10'
mpl.rcParams['font.size'] = 13

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, ticker
from os.path import expanduser, join

from cogspaces.datasets.utils import get_output_dir
from exps.grids.gather_quantitative import get_chance_subjects


import matplotlib.pyplot as plt

idx = pd.IndexSlice

save_dir = '/home/arthur/work/papers/papers/2018_05_nature/figures/'

pad_bottom = .47
pad_top = .02
pad_right = .02
pad_left = 2.49


def make_data():

    output_dir = get_output_dir()

    factored_output_dir = join(output_dir,
                               'factored_refit_gm_normal_init_rest_positive_notune')
    baseline_output_dir = join(output_dir, 'full_logistic')

    factored = pd.read_pickle(join(factored_output_dir, 'accuracies.pkl'))
    factored = factored.loc[0.0001]

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
    width, height = 9, 5.6

    with open(expanduser('~/work/repos/cogspaces/cogspaces/'
                         'datasets/brainpedia.json')) as f:
        names = json.load(f)

    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[2.7, 1]
                           )
    fig = plt.figure(figsize=(width, height))
    gs.update(left=2.2 / width, right=1 - .2 / width,
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
    ax1.set_xlabel('Accuracy gain')
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
    ax2.set_xlabel('Decoding accuracy on test set')

    handles = [rects1, rects2, lines]
    labels = ['Baseline decoder', 'Factored decoder\nwith multi-study\nprior',
              'Chance level']
    ax2.legend(handles, labels, loc='lower right', frameon=False,
               fontsize=11,
               bbox_to_anchor=(1.14, .64))

    ax2.set_yticks(ind + width / 2)
    ax2.annotate('Task fMRI study', xy=(-.4, -.08), xycoords='axes fraction')
    labels = [
        '%s' % (names[label]['title']) if label in names else label
        for label in data.index.values]
    ax2.set_yticklabels(labels, ha='right',
                        va='center')
    sns.despine(fig)

    plt.savefig(join(save_dir, 'joined_mean.pdf'), facecolor=None, edgecolor=None,
                transparent=True)
    plt.close(fig)

    return sort


def plot_compare_methods(sort, many=False):
    width, height = 6.2, 3

    output_dir = get_output_dir()

    baseline = 'full_logistic'

    if many:
        exps = [baseline,
                'logistic_gm',
                # 'factored_refit_gm_rest_positive_notune',
                # 'factored_refit_gm_notune',
                # 'factored_gm',
                # 'factored_refit_gm_normal_init_rest_positive_notune',
                # # 'factored_refit_gm_normal_init_positive_notune',
                # 'factored_refit_gm_normal_init_notune',
                # 'factored_gm_normal_init',
                ]
    else:
        exps = [baseline, 'logistic_gm',
                'factored_refit_gm_rest_positive_notune']

    dfs = []
    for exp in exps:
        df = pd.read_pickle(join(output_dir, exp, 'accuracies.pkl'))
        if 'refit' in exp:
            if exp == 'factored_refit_gm_rest_positive_notune':
                df = df.loc[0.0001]
            else:
                df = df.loc[0.0001]
        if exp in ['factored_gm', 'factored_gm_normal_init']:
            df = df.groupby(['study', 'seed']).mean()
        dfs.append(df)

    df = pd.concat(dfs, axis=0, keys=exps, names=['method'])

    df_std = []
    methods = []
    for method, sub_df in df.groupby('method'):
        methods.append(method)
        df_std.append(
            sub_df.loc[method] - df.loc[baseline].groupby('study').transform(
                'median'))
    df_std = pd.concat(df_std, keys=methods, names=['method'])

    sort = pd.MultiIndex.from_product([methods, sort],
                                      names=['method', 'study'])
    df_sort = df_std.reset_index('seed').loc[sort]

    df_sort = df_sort.reset_index()

    n_studies = len(df_sort['study'].unique())

    diff_color = sns.color_palette("husl", n_studies)
    if many:
        height = height * len(exps) / 3

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    fig.subplots_adjust(left=.07 / width, right=1 - 1.6 /width,
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
    ax.set_xlabel('Accuracy gain compared to baseline median')
    ax.set_ylabel('')
    ax.spines['left'].set_position('zero')
    plt.setp(ax.spines['left'], zorder=2)
    ax.set_yticklabels([])
    ax.set_yticks([])

    methods = df_sort['method'].unique()

    y_labels = {}
    y_labels['factored_gm_single'] = (
        'Factored decoder\nwith single-\nstudy prior'
    )
    y_labels[baseline] = ('Decoding from\nvoxels')
    y_labels['logistic_gm'] = (
        'Decoding from rest\n'
        'functional networks'
    )
    y_labels['factored_refit_gm_rest_positive_notune'] = (
        'Decoding from\n'
        'multi-study\n'
        'end-to-end trained\n'
        'task networks'
    )

    if many:
        y_labels['factored_refit_gm_normal_init_rest_positive_notune'] = (
            'Random init +\n'
            'sparse NMF\nwith rest init')
        y_labels['factored_refit_gm_normal_init_positive_notune'] = (
            'Random init +\n'
            'sparse NMF\nwith random init')
        y_labels['factored_refit_gm_normal_init_notune'] = (
            'Random init +\n'
            'DL with\nrandom init')
        y_labels['factored_gm_normal_init'] = (
            'Random init')
        y_labels['factored_refit_gm_rest_positive_notune'] = (
            'Rest init +\n'
            'sparse NMF\n'
            'with rest init')
        y_labels['factored_refit_gm_notune'] = (
            'Rest init +\n'
            'DL with rest init')
        y_labels['factored_gm'] = (
            'Rest init')
        y_labels['full_logistic'] = 'Full logistic'
        y_labels['logistic_gm'] = 'Rest compressed logistic'

        ax.hlines([1.5, 3.5], *ax.get_xlim(), linestyle='--', color='.5')
        ax.annotate('Ablation',
                    xy=(-.12, 2.5), xytext=(-7, 0),
                    textcoords="offset points",
                    fontsize=13,
                    xycoords='data',
                    va='center', rotation=90,
                    ha='right')
        ax.annotate('Random init',
                    xy=(-.12, 5), xytext=(-7, 0),
                    textcoords="offset points",
                    fontsize=13,
                    xycoords='data',
                    va='center', rotation=90,
                    ha='right')
    for i, method in enumerate(methods):
        ax.annotate(y_labels[method], xy=(.17, i), xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=13,
                    xycoords='data',
                    va='center',
                    ha='left')

    sns.despine(fig)

    plt.setp(ax.legend(), visible=False)
    plt.savefig(join(save_dir, 'comparison_method%s.pdf' %
                     ('_many' if many else '')))
    plt.close(fig)


def plot_gain_vs_size(sort):
    width, height = 3, 2.3

    output_dir = get_output_dir()

    colors = sns.color_palette('husl', len(sort))
    colors = {study: color for study, color in zip(sort, colors)}

    factored_output_dir = join(output_dir, 'factored_refit_gm_notune')
    baseline_output_dir = join(output_dir, 'logistic_gm')

    factored = pd.read_pickle(join(factored_output_dir, 'accuracies.pkl'))
    factored = factored.loc[0.0001]

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

    fig, ax = plt.subplots(1, 1, figsize=(width, height),)

    sns.regplot(joined_mean['n_subjects', 'mean'], joined_mean['diff', 'mean'],
                n_boot=1000, ax=ax, color='black', logx=True,
                order=1, truncate=True, lowess=True,
                scatter=False)
    # xdata = ax.lines[0].get_xdata()
    # ax.lines[0].set_xdata(np.exp(xdata))
    # vertices = ax.collections[0].get_paths()[0].vertices
    # vertices[:, 0] = np.exp(vertices[:, 0])
    # print(joined_mean['n_subjects', 'mean'])
    clouds = ax.scatter(joined_mean['n_subjects', 'mean'], joined_mean['diff', 'mean'],
               c=list(map(lambda x: colors[x],
                          joined_mean.index.get_level_values('study'))),
               s=8, )
    ax.set_xscale('log')
    ax.set_ylabel('Accuracy gain')
    ax.set_xlabel('Number of train subjects')
    ax.set_xlim([3, 420])
    ax.set_xticks([4, 16, 32, 100, 300, 400])
    ax.set_xticklabels([4, 16, 32, 100, 400])

    ax.legend([clouds], ['Study'], frameon=True, scatterpoints=3, fontsize=11)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.spines['bottom'].set_position('zero')

    fig.subplots_adjust(left=.7 / width, right=1 - pad_right / width,
                        bottom=0.3 / height, top=1 - .08 / height, )

    sns.despine(fig)
    fig.savefig(join(save_dir, 'gain_vs_size.pdf'))


def plot_gain_vs_accuracy(sort):
    width, height = 3, 2.3

    metrics = pd.read_pickle(join(get_output_dir(), 'factored_refit_gm_low_lr',
                                  'metrics.pkl'))
    metrics = metrics.loc[0.0001]
    ref_metrics = pd.read_pickle(join(get_output_dir(), 'logistic_gm',
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

        fig, ax = plt.subplots(1, 1, figsize=(width, height),)
        fig.subplots_adjust(left=.7 / width, right=1 - .2 / width,
                            bottom=pad_bottom / height, top=1 - .08 / height, )

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
            ax.set_xlabel('Baseline balanced accuracy score \n(per contrast)')
        else:
            ax.set_xlabel('Baseline F1 score (per contrast)')
        ax.set_ylabel('Gain from multi-study model     ')
        ax.hlines(0, 0, 1, linestyle=(0, [2, 4]))

        ax.legend([clouds], ['Contrast'], frameon=True, scatterpoints=3,
                  fontsize=11,
                  loc='lower right')

        sns.despine(fig)
        fig.savefig(join(save_dir, 'gain_vs_accuracy_%s.pdf' % score))


if __name__ == '__main__':
    data, sort = make_data()
    plot_joined(data)
    plot_compare_methods(sort)
    plot_compare_methods(sort, many=True)
    plot_gain_vs_accuracy(sort)
    plot_gain_vs_size(sort)
