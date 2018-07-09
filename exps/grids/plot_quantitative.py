import json
import matplotlib as mpl

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

save_dir = '/home/arthur/work/papers/papers/thesis/figures/nature/quantitative'

pad_bottom = .51
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
    width, height = 8.5, 5.6

    with open(expanduser('~/work/repos/cogspaces/cogspaces/'
                         'datasets/brainpedia.json')) as f:
        names = json.load(f)

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
        '%s' % (names[label]['title']) if label in names else label
        for label in data.index.values]
    ax2.set_yticklabels(labels, ha='right', va='center', fontsize=8.5)
    sns.despine(fig)

    plt.savefig(join(save_dir, 'joined_mean.pdf'), facecolor=None,
                edgecolor=None,
                transparent=True)
    plt.close(fig)

    return sort


def plot_compare_methods(sort, ablation=None):
    width, height = 6.2, 3

    # if ablation is not None:
    #     width = 8

    output_dir = get_output_dir()

    if ablation is None:
        baseline = 'full_logistic'
    elif ablation == 'posthoc':
        baseline = 'factored_refit_gm_rest_positive_notune'
    elif ablation == 'gm':
        baseline = 'factored_gm'
    elif ablation == 'transfer':
        baseline = 'factored_gm'
    elif ablation == 'dropout':
        baseline = 'factored_gm'
    elif ablation == 'l2':
        baseline = 'factored_gm'


    if ablation is None:
        exps = [baseline, 'logistic_gm',
                'factored_refit_gm_rest_positive_notune']
    elif ablation == 'gm':
        exps = ['full_logistic', baseline,
                'factored']
    elif ablation == 'posthoc':
        exps = ['full_logistic', 'factored_gm_normal_init',
                'factored_gm',
                'factored_refit_gm_normal_init_positive_notune',
                baseline,
                ]
    elif ablation == 'dropout':
        exps = ['full_logistic',
                'adaptive_dropout',
                'bn',
                baseline,
                ]
    elif ablation == 'l2':
        exps = ['full_logistic',
                'factored_l2',
                baseline,
                ]
    elif ablation == 'transfer':
        exps = ['full_logistic', 'factored_transfer', baseline]
    else:
        raise ValueError

    dfs = []
    for exp in exps:
        df = pd.read_pickle(join(output_dir, exp, 'accuracies.pkl'))
        if 'refit' in exp:
            if exp == 'factored_refit_gm_rest_positive_notune':
                df = df.loc[0.0001]
            else:
                df = df.loc[0.001]
        if exp in ['factored_gm', 'factored_gm_normal_init', 'factored']:
            df = df.groupby(['study', 'seed']).mean()
        if exp == 'factored_l2':
            df = df.groupby(['study', 'seed']).max()
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
    if ablation is not None:
        height = height * len(exps) / 3

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    if ablation is None:
        fig.subplots_adjust(left=.07 / width, right=1 - 1.65 / width,
                            bottom=0.55 / height, top=1 - pad_top / height, )
    else:
        fig.subplots_adjust(left=.11 / width, right=1 - 1.8 / width,
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
    if ablation is None:
        ax.set_xlim([-0.14, 0.175])
    else:
        ax.set_xlim([-0.125, 0.11])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    if ablation is None:
        ax.set_xlabel('Accuracy gain compared to baseline median', fontsize=15)
    else:
        ax.set_xlabel('Accuracy gain compared to proposed model median',
                      fontsize=15)

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
    y_labels['full_logistic'] = ('Decoding from\nvoxels')
    y_labels['logistic_gm'] = (
        'Decoding from\n'
        'rest networks'
    )
    y_labels['factored_refit_gm_rest_positive_notune'] = (
        'Decoding from\n'
        'multi-study\n'
        'task-optimized\n'
        'networks'
    )

    if ablation is not None:
        if ablation == 'init':
            y_labels['factored_refit_gm_rest_positive_notune'] = (
                'Resting-state init.\nfor second layer')
        elif ablation == 'posthoc':
            y_labels['factored_refit_gm_rest_positive_notune'] = (
                # 'Proposed model:\n'
                'Resting-state init.\npost-hoc transform.')
        elif ablation == 'gm':
            y_labels['factored_refit_gm_rest_positive_notune'] = (
                # 'Proposed model:\n'
                'Decoding from\n'
                'Grey matter\n'
                'func networks')
        y_labels['full_logistic'] = (
            # 'Baseline:\n'
            'Decoding from\nvoxels')

        y_labels['factored_refit_gm_normal_init_positive_notune'] = (
            'Random init.\nfor second layer')
        y_labels['factored_refit_gm_normal_init_notune'] = (
            'Random init +\n'
            'DL with\nrandom init')
        y_labels['factored_gm_normal_init'] = (
            'Random init. and\n'
            'no post-hoc\ntransform.')
        y_labels['factored_refit_gm_notune'] = (
            'Rest init +\n'
            'DL with rest init')
        if ablation == 'posthoc':
            y_labels['factored_gm'] = (
                'Training without\npost-hoc transform.')
        elif ablation == 'gm':
            y_labels['factored_gm'] = (
                # 'Proposed model:\n'
                'Grey matter\n'
                'func. networks')
        elif ablation == 'transfer':
            y_labels['factored_gm'] = (
                # 'Proposed model:\n'
                'Joint training\n'
                'of all studies')
        elif ablation == 'dropout':
            y_labels['factored_gm'] = (
                # 'Proposed model:\n'
                'Adaptive dropout\n'
                'and batch norm.')
        elif ablation == 'l2':
            y_labels['factored_gm'] = (
                # 'Proposed model:\n'
                'Transfer via\n'
                'dropout +\n'
                'hard rank constr.')
        y_labels['factored_transfer'] = (
            'Decoding from\n'
            'pre-learned\n'
            'second layer   ')
        y_labels['factored'] = (
            'Decoding from\n'
            'all-brain\n'
            'func. networks')
        y_labels['factored_l2'] = (
            'Transfer via $\\ell_2$\n'
            'regularization')
        y_labels['bn'] = (
            'No batch norm.')
        y_labels['adaptive_dropout'] = (
            'Fixed dropout')
        y_labels['logistic_gm'] = 'Rest compressed logistic'

        if ablation in ['gm', 'transfer', 'l2']:
            line = ax.axhline(0.5, -.2, 1.4, linestyle='--', color='.5')
            line.set_clip_on(False)
            y_text = 1
            line = ax.axhline(1.5, -.2, 1.4, linestyle='--', color='.5')
            line.set_clip_on(False)

        elif ablation == 'posthoc':
            y_text = 2
            line = ax.axhline(0.5, -.2, 1.4, linestyle='--', color='.5')
            line.set_clip_on(False)
            line = ax.axhline(3.5, -.2, 1.4, linestyle='--', color='.5')
            line.set_clip_on(False)
        elif ablation == 'dropout':
            y_text = 1.5
            line = ax.axhline(0.5, -.2, 1.4, linestyle='--', color='.5')
            line.set_clip_on(False)
            line = ax.axhline(2.5, -.2, 1.4, linestyle='--', color='.5')
            line.set_clip_on(False)

        if ablation is not None:
            props = dict(xytext=(-2, 0),
                         textcoords="offset points",
                         bbox={'facecolor': 'black',
                               'boxstyle': 'round',
                               'linewidth': 0},
                         color='white',
                         fontsize=13,
                         xycoords='data',
                         va='center', rotation=0,
                         zorder=300,
                         ha='right')

            ax.annotate('Ablation',
                        xy=(-.09, y_text), **props)
            ax.annotate('Baseline',
                        xy=(-.092, 0.25), **props)
            ax.annotate('Proposed',
                        xy=(-.088, len(exps) - 1), **props)
        # ax.annotate('Random init',
        #             xy=(-.12, 5), xytext=(-7, 0),
        #             textcoords="offset points",
        #             fontsize=15,
        #             xycoords='data',
        #             va='center', rotation=90,
        #             ha='right')
    for i, method in enumerate(methods):
        ax.annotate(y_labels[method], xy=(.11, i), xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=15,
                    xycoords='data',
                    va='center',
                    ha='left')

    sns.despine(fig)

    plt.setp(ax.legend(), visible=False)
    plt.savefig(join(save_dir, 'comparison_method%s.pdf' % (
        '_%s' % ablation if ablation is not None else '')))
    plt.close(fig)


def plot_gain_vs_size(sort):
    width, height = 3, 2.3

    output_dir = get_output_dir()

    colors = sns.color_palette('husl', len(sort))
    colors = {study: color for study, color in zip(sort, colors)}

    factored_output_dir = join(output_dir, 'factored_refit_gm_notune')
    baseline_output_dir = join(output_dir, 'full_logistic')

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
    ax.set_xlabel('Number of train subjects', fontsize=15)
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


def plot_gain_vs_size_multi():
    width, height = 3, 2.3

    output_dir = get_output_dir()

    factored_output_dir = join(output_dir, 'weight_power')
    baseline_output_dir = join(output_dir, 'full_logistic')

    many_factored = pd.read_pickle(join(factored_output_dir, 'accuracies.pkl'))
    baseline = pd.read_pickle(join(baseline_output_dir, 'accuracies.pkl'))
    chance_level, n_subjects = get_chance_subjects()

    handles = []
    labels = []
    fig, ax = plt.subplots(1, 1, figsize=(width, height), )

    for weight_power in [0., 6 / 9, 1.]:
        factored = many_factored.loc[weight_power]
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

        sns.regplot(joined_mean['n_subjects', 'mean'], joined_mean['diff', 'mean'],
                    n_boot=10, ax=ax, logx=True,
                    order=1, truncate=True, lowess=True,
                    scatter=False)
        handles.append(ax.scatter(joined_mean['n_subjects', 'mean'],
                            joined_mean['diff', 'mean'],
                            s=4, ))
        labels.append(weight_power)
    ax.set_xscale('log')
    ax.set_ylabel('Multi study acc. gain', fontsize=15)
    ax.set_xlabel('Number of train subjects', fontsize=15)
    ax.set_xlim([3, 420])
    ax.set_xticks([4, 16, 32, 100, 300, 400])
    ax.set_xticklabels([4, 16, 32, 100, 400])
    ax.set_ylim([-0.075, 0.15])

    labels = ['$\\beta = %.1f$' % beta for beta in [0., 6/9, 1]]

    ax.legend(handles, labels, frameon=True, scatterpoints=3, fontsize=10)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.spines['bottom'].set_position('zero')

    fig.subplots_adjust(left=.75 / width, right=1 - pad_right / width,
                        bottom=0.35 / height, top=1 - .08 / height, )

    sns.despine(fig)
    fig.savefig(join(save_dir, 'gain_vs_size_multi.pdf'))


def plot_weight_power():
    width, height = 3, 2.3

    output_dir = get_output_dir()

    factored_output_dir = join(output_dir, 'weight_power')
    baseline_output_dir = join(output_dir, 'full_logistic')

    many_factored = pd.read_pickle(join(factored_output_dir, 'accuracies.pkl'))
    baseline = pd.read_pickle(join(baseline_output_dir, 'accuracies.pkl'))
    fig, ax = plt.subplots(1, 1, figsize=(width, height), constrained_layout=True)

    baseline = pd.concat([baseline] * 10, keys=np.linspace(0, 1, 10),
                         names=['weight_power'])
    joined = pd.concat([many_factored, baseline],
                       keys=['factored', 'baseline'], axis=1)
    joined['diff'] = joined['factored'] - joined['baseline']
    data = joined['diff'].reset_index()
    data = data.dropna()
    print(data)
    # sns.boxplot(x='weight_power', y='diff', data=data, ax=ax, whis=0, showfliers=False,
    #             color='grey')
    ax.set_ylim([-0.0, 0.07])
    res = joined['diff'].groupby(level='weight_power').aggregate({'median': lambda x: x.median(),
                                                                  '25quart': lambda x: x.quantile(.25),
                                                                  '75quart': lambda x: x.quantile(.75)})
    ax.plot(np.linspace(0, 1, 10), res['median'])
    ax.fill_between(np.linspace(0, 1, 10), res['25quart'], res['75quart'], alpha=0.25)
    ax.yaxis.set_major_formatter(
        ticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.025))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.0125))
    ax.set_xlabel('Study weight $\\sim$ size$^\\beta$')
    ax.set_ylabel('Accuracy gain (median)')


    sns.despine(fig)
    fig.savefig(join(save_dir, 'weight_power.pdf'))


def plot_gain_vs_accuracy(sort):
    width, height = 3, 2.3

    metrics = pd.read_pickle(join(get_output_dir(), 'factored_refit_gm_low_lr',
                                  'metrics.pkl'))
    metrics = metrics.loc[0.0001]
    ref_metrics = pd.read_pickle(join(get_output_dir(), 'full_logistic',
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
            ax.set_xlabel('Baseline balanced accuracy',
                          fontsize=15)
        else:
            ax.set_xlabel('Baseline F1 score (per contrast)')
        ax.set_ylabel('Multi-study b-acc. gain    ', fontsize=15)
        ax.hlines(0, 0, 1, linestyle=(0, [2, 4]))

        ax.legend([clouds], ['Contrast'], frameon=True, scatterpoints=3,
                  fontsize=13,
                  loc='lower right')

        sns.despine(fig)
        fig.savefig(join(save_dir, 'gain_vs_accuracy_%s.pdf' % score))


if __name__ == '__main__':
    data, sort = make_data()
    # plot_joined(data)
    # plot_compare_methods(sort)
    # plot_compare_methods(sort, ablation='posthoc')
    # plot_compare_methods(sort, ablation='gm')
    # plot_compare_methods(sort, ablation='transfer')
    # plot_compare_methods(sort, ablation='dropout')
    # plot_compare_methods(sort, ablation='l2')
    # plot_gain_vs_accuracy(sort)
    # plot_gain_vs_size(sort)
    plot_gain_vs_size_multi()
    plot_weight_power()