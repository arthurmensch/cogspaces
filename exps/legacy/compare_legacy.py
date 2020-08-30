import matplotlib as mpl
import matplotlib.pyplot as plt

# plt.rc('text', usetex=True)
# font = {'family': 'sans-serif', 'size': 13,
#         'sans-serif': ['computer modern sans']}
# plt.rc('font', **font)
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
from cogspaces.datasets.derivative import get_study_info

mpl.rcParams['font.family'] = 'CMU Sans Serif'
mpl.rcParams['font.size'] = 13

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, ticker
from os.path import join
import os

from cogspaces.datasets.utils import get_output_dir

idx = pd.IndexSlice

save_dir = join(get_output_dir(), 'revision_output')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pad_bottom = .51
pad_top = .02
pad_right = .02
pad_left = 2.49


def make_data(sort_by_chance=False):
    output_dir = get_output_dir()

    factored_output_dir = join(output_dir,
                               'factored_refit_gm_rest_positive_notune')
    baseline_output_dir = join(output_dir, 'full_logistic')

    factored = pd.read_pickle(join(factored_output_dir, 'accuracies.pkl'))
    factored = factored.loc[0.0001]

    baseline = pd.read_pickle(join(baseline_output_dir, 'accuracies.pkl'))

    joined = pd.concat([factored, baseline],
                       keys=['factored', 'baseline'], axis=1)
    joined['diff'] = joined['factored'] - joined['baseline']
    joined_mean = joined.groupby('study').aggregate(['mean', 'std'])

    data = joined_mean
    data = data.sort_values(('diff', 'mean'), ascending=True)

    info = get_study_info().groupby(by='study')[['chance_study', 'name_study', 'latex_name_study']].first()
    data = data.join(info)

    if sort_by_chance:
        data = data.sort_values('chance_study', ascending=True)
    sort = data.index.values.tolist()[::-1]
    return data, sort


def plot_joined(data, sort_by_chance=False):
    width, height = 8.5, 5.6

    # with open(expanduser('~/work/repos/cogspaces/cogspaces/'
    #                      'datasets/brainpedia.json')) as f:
    #     names = json.load(f)
    df = pd.read_csv('~/work/papers/papers/thesis/brainpedia.csv', index_col=0,
                     header=0)
    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[2.7, 1]
                           )
    fig = plt.figure(figsize=(width, height))
    gs.update(left=2.5 / width, right=1 - .2 / width,
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

    lines = ax2.vlines([data['chance_study']],
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
    ax2.set_yticklabels(data['name_study'], ha='right', va='center', fontsize=8.5)
    sns.despine(fig)

    plt.savefig(join(save_dir, f'joined_mean{"_chance" if sort_by_chance else ""}.pdf'), facecolor=None,
                edgecolor=None,
                transparent=True)
    plt.close(fig)


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
    elif ablation == 'latent_size':
        baseline = 'full_logistic'

    if ablation is None:
        exps = [baseline, 'logistic_gm',
                'factored_refit_gm_rest_positive_notune']
    elif ablation == 'gm':
        exps = ['full_logistic', 'factored', baseline
                ]
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
                'factored_l2_no_rank',
                'factored_l2_good',
                baseline,
                ]
    elif ablation == 'transfer':
        exps = ['full_logistic', 'factored_transfer', baseline]
    elif ablation == 'latent_size':
        exps = ['latent_size_2', baseline]
    else:
        raise ValueError

    dfs = []
    for exp in exps:
        df = pd.read_pickle(join(output_dir, exp, 'accuracies.pkl')).to_frame()
        if 'refit' in exp:
            if exp == 'factored_refit_gm_rest_positive_notune':
                df = df.loc[0.0001]
            else:
                df = df.loc[0.00001]
        if exp in ['factored_gm', 'factored_gm_normal_init', 'factored']:
            df = df.groupby(['study', 'seed']).mean()
        elif exp in ['factored_l2_good', 'factored_l2_no_rank', 'factored_l2']:
            df = df.groupby(['study', 'seed']).max()
        elif exp == 'latent_size_2':
            df = df.reset_index().rename(columns={'latent_size': 'method'})
        if ablation == 'latent_size' and exp == baseline:
            df = df.reset_index()
            df['method'] = baseline

        dfs.append(df)

    if ablation == 'latent_size':
        df = pd.concat(dfs, axis=0).set_index(['method', 'study', 'seed'])
        df = df.rename(index={'auto': 545})
        df = df.sort_index(level='method')
    else:
        df = pd.concat(dfs, axis=0, keys=exps, names=['method'])

    df_std = []
    df_mean = []
    crossed_methods = []
    methods = df.index.get_level_values('method').unique()
    for i, method in enumerate(methods):
        df_std.append(
            df.loc[method] - df.loc[baseline].groupby('study').transform(
                'median'))
        for method_ref in [baseline]:
            crossed_methods.append(str(method) + '-' + str(method_ref))
            diff = df.loc[method] - df.loc[method_ref]
            df_mean.append(diff.agg(['mean', 'std', 'median',
                                     lambda x: (x >= 0).mean()]))
    df_std = pd.concat(df_std, keys=methods, names=['method'])
    df_mean = pd.concat(df_mean, keys=crossed_methods, names=['method'])
    print(df_mean)

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
        elif ablation == 'latent_size':
            height *= 2
            width *= 1.3

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    if ablation is None:
        fig.subplots_adjust(left=.07 / width, right=1 - 1.9 / width,
                            bottom=0.55 / height, top=1 - pad_top / height, )
    elif ablation == 'latent_size':
        fig.subplots_adjust(left=.07 / width, right=1 - 2.3 / width,
                            bottom=0.55 / height, top=1 - pad_top / height, )
    else:
        fig.subplots_adjust(left=.2 / width, right=1 - 1.9 / width,
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
    elif ablation == 'latent_size':
        ax.set_xlim([-0.125, 0.11])
    else:
        ax.set_xlim([-0.125, 0.11])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    if ablation is None or ablation == 'latent_size':
        ax.set_xlabel('Accuracy gain compared to voxel-based decoding', fontsize=15)
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
    y_labels['full_logistic'] = ('Decoding from:\nVoxels')
    y_labels['logistic_gm'] = (
        'Resting-state\n'
        'functional units')
    y_labels['factored_refit_gm_rest_positive_notune'] = (
        'Multi-study\n'
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
                'Resting-state init.\nConsensus model')
        elif ablation == 'gm':
            y_labels['factored_refit_gm_rest_positive_notune'] = (
                # 'Proposed model:\n'
                'Decoding from\n'
                'Grey matter\n'
                'func networks')
        y_labels['full_logistic'] = (
            'Standard decoding:\nfrom voxels')

        y_labels['factored_refit_gm_normal_init_positive_notune'] = (
            'Random init.\nConsensus model')
        y_labels['factored_refit_gm_normal_init_notune'] = (
            'Random init +\n'
            'DL with\nrandom init')
        y_labels['factored_gm_normal_init'] = (
            'Random init.\n'
            'No consensus')
        y_labels['factored_refit_gm_notune'] = (
            'Resting-state init +\n'
            'DL with rest init')
        if ablation == 'posthoc':
            y_labels['factored_gm'] = (
                'Resting-state init.\n'
                'No consensus')
        elif ablation == 'gm':
            y_labels['factored_gm'] = (
                # 'Proposed model:\n'
                'Grey matter\n'
                'func. networks')
        elif ablation == 'transfer':
            y_labels['factored_gm'] = (
                # 'Proposed model:\n'
                '2$^{nd}$ + 3$^{rd}$ layer\ntrained '
                'on\nN studies jointly')
        elif ablation == 'dropout':
            y_labels['factored_gm'] = (
                # 'Proposed model:\n'
                'Adaptive Dropout\n'
                'and batch norm.')
        elif ablation == 'l2':
            y_labels['factored_gm'] = (
                # 'Proposed model:\n'
                'Transfer via\n'
                'Dropout +\n'
                'hard rank constr.')
        y_labels['factored_transfer'] = (
            '2$^{nd}$ layer trained\non '
            'N - 1 studies\n'
            '3$^{rd}$ layer trained\non '
            'target study')
        y_labels['factored'] = (
            'Decoding from\n'
            'all-brain\n'
            'func. networks')
        y_labels['factored_l2_good'] = (
            'Transfer via $\\ell_2$\n'
            'regularization +\n'
            'hard rank constr.')
        y_labels['factored_l2_no_rank'] = (
            'Transfer via $\\ell_2$\n'
            'regularization')
        y_labels['bn'] = (
            'No batch norm.')
        y_labels['adaptive_dropout'] = (
            'Fixed Dropout')
        y_labels['logistic_gm'] = 'Rest compressed logistic'

        if ablation == 'latent_size':
            for method in methods:
                if method == baseline:
                    y_labels[method] = 'Voxel-based decoding'
                else:
                    y_labels[method] = f'$l={method}$ components'

        if ablation in ['gm', 'transfer']:
            line = ax.axhline(0.5, -.2, 1.5, linestyle='--', color='.5')
            line.set_clip_on(False)
            y_text = 1
            line = ax.axhline(1.5, -.2, 1.5, linestyle=':', color='.6')
            line.set_clip_on(False)

        elif ablation == 'posthoc':
            y_text = 2
            line = ax.axhline(0.5, -.2, 1.5, linestyle='--', color='.5')
            line.set_clip_on(False)
            line = ax.axhline(3.5, -.2, 1.5, linestyle=':', color='.6')
            line.set_clip_on(False)
        elif ablation in ['dropout', 'l2']:
            y_text = 1.5
            line = ax.axhline(0.5, -.2, 1.5, linestyle='--', color='.5')
            line.set_clip_on(False)
            line = ax.axhline(2.5, -.2, 1.5, linestyle=':', color='.6')
            line.set_clip_on(False)
        elif ablation == 'latent_size':
            line = ax.axhline(4.5, -.2, 1.5, linestyle='--', color='.5')
            line.set_clip_on(False)
        else:
            y_text = 1
        # ax.annotate('Random init',
        #             xy=(-.12, 5), xytext=(-7, 0),
        #             textcoords="offset points",
        #             fontsize=15,
        #             xycoords='data',
        #             va='center', rotation=90,
        #             ha='right')
    for i, method in enumerate(methods):
        if i == len(methods) - 1:
            fontweight = 'bold'
        else:
            fontweight = 'normal'
            # y_labels[method] = r'\textbf{' + y_labels[method].replace('\n', '\\linebreak ') + r'}'
        # font = FontProperties(weight=fontweight, size=15,)
        ax.annotate(y_labels[method], xy=(1, i), xytext=(60, 0),
                    textcoords="offset points",
                    fontsize=15 if method != 'factored_transfer' else 13,
                    fontweight=fontweight,
                    xycoords=('axes fraction', 'data'),
                    va='center',
                    ha='center')
    # if ablation is None:
    #     ax.annotate('Decoding from:', xy=(1, -.5), xytext=(60, 0),
    #                 textcoords="offset points",
    #                 fontsize=15 if method != 'factored_transfer' else 13,
    #                 fontweight=fontweight,
    #                 xycoords=('axes fraction', 'data'),
    #                 va='center',
    #                 ha='center')
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

        sns.regplot(joined_mean['n_subjects', 'mean'],
                    joined_mean['diff', 'mean'],
                    n_boot=10, ax=ax, logx=True,
                    order=1, truncate=True, lowess=True,
                    scatter=False)
        handles.append(ax.scatter(joined_mean['n_subjects', 'mean'],
                                  joined_mean['diff', 'mean'],
                                  s=4, ))
        labels.append(weight_power)
    ax.set_xscale('log')
    ax.set_ylabel('Multi study acc. gain', fontsize=15)
    ax.set_xlabel('Number of train subjects', fontsize=13, zorder=400)
    ax.set_xlim([3, 420])
    ax.set_xticks([4, 16, 32, 100, 300, 400])
    ax.set_xticklabels([4, 16, 32, 100, 400])
    ax.set_ylim([-0.075, 0.15])

    labels = ['$\\beta = %.1f$' % beta for beta in [0., .6, 1]]

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
    fig, ax = plt.subplots(1, 1, figsize=(width, height),
                           constrained_layout=True)

    baseline = pd.concat([baseline] * 10, keys=np.linspace(0, 1, 10),
                         names=['weight_power'])
    joined = pd.concat([many_factored, baseline],
                       keys=['factored', 'baseline'], axis=1)
    joined['diff'] = joined['factored'] - joined['baseline']
    data = joined['diff'].reset_index()
    data = data.dropna()
    # sns.boxplot(x='weight_power', y='diff', data=data, ax=ax, whis=0, showfliers=False,
    #             color='grey')
    ax.set_ylim([-0.0, 0.07])
    res = joined['diff'].groupby(level='weight_power').aggregate(
        {'median': lambda x: x.median(),
         '25quart': lambda x: x.quantile(.25),
         '75quart': lambda x: x.quantile(.75)})
    ax.plot(np.linspace(0, 1, 10), res['median'])
    ax.fill_between(np.linspace(0, 1, 10), res['25quart'], res['75quart'],
                    alpha=0.25)
    ax.yaxis.set_major_formatter(
        ticker.PercentFormatter(xmax=1, decimals=1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.025))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.0125))
    ax.set_xlabel('$\\beta$ s.t. study weight $\\sim$ size$^\\beta$')
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


def write_latex(data):
    data = data.sort_index().reset_index()
    for x in ['factored', 'baseline', 'diff']:
        data[x] = data.apply(lambda r: f'${r[(x, "mean")] * 100:.0f}\pm{r[(x, "std")] * 100:.0f}\%$',
                             axis='columns')
    data['chance_study'] = data['chance_study'].map(lambda x: f'${x * 100:.0f}\%$')
    table_data = data[['latex_name_study', 'chance_study', 'factored', 'baseline', 'diff']]
    table_data.columns = pd.Index(
        ['Study', 'Chance level', 'Multi-task accuracy', 'Single-task accuracy', 'Accuracy gain'])
    table_data = table_data.reset_index(drop=True).set_index(['Study'])
    with pd.option_context("max_colwidth", 1000):
        latex = table_data.to_latex(sparsify=True, escape=False, column_format='p{4cm}p{1cm}p{1cm}p{1cm}p{1cm}')
    with open(join(save_dir, 'accuracies.tex'), 'w+') as f:
        f.write(latex)


def write_contrast_latex():
    metrics = pd.read_pickle(join(get_output_dir(), 'factored_refit_gm_rest_positive_notune',
                                  'metrics.pkl'))
    metrics = metrics.loc[0.0001]
    ref_metrics = pd.read_pickle(join(get_output_dir(), 'full_logistic',
                                      'metrics.pkl'))
    joined = pd.concat([metrics, ref_metrics], axis=1,
                       keys=['factored', 'baseline'], join='inner')
    diff = joined['factored'] - joined['baseline']
    for v in diff.columns:
        joined['diff', v] = diff[v]
    joined = joined.reset_index().groupby(by=['study', 'contrast']).aggregate(['mean', 'std']).reset_index()
    joined['contrast'] = joined['contrast'].apply(lambda x: x.replace('_', ' ').lower())
    data = get_study_info().merge(joined, how='left', on=['study', 'contrast'])

    for x in ['factored', 'baseline', 'diff']:
        data[x] = data.apply(
            lambda r: f'${r[(x, "bacc", "mean")] * 100:.0f}\pm{r[(x, "bacc", "std")] * 100:.0f}\%$',
            axis='columns')
    data = data[['latex_cite', 'task', 'contrast', 'factored', 'baseline', 'diff']]

    data['contrast'] = data['contrast'].apply(lambda x: x.replace('%', '\%').replace('&', '\&'))
    data['task'] = data['task'].apply(lambda x: x.replace('%', '\%').replace('&', '\&'))

    data.columns = pd.Index(
        ['Study', 'Task', 'Contrast', 'Multi-study B-acc', 'Voxel-level B-acc', 'B-acc gain'])
    data = data.reset_index(drop=True).set_index(['Study', 'Task', 'Contrast'])
    with pd.option_context("max_colwidth", 1000):
        latex = data.to_latex(sparsify=True, longtable=True, escape=False, column_format='p{0.8cm}p{2cm}p{3.5cm}p{1.5cm}p{1cm}p{1cm}',
                              caption="List of all contrasts used in this paper, per study of origin and task.",
                              label="table:all_contrasts")
    with open(join(save_dir, 'contrast_accuracies.tex'), 'w+') as f:
        f.write(latex)


if __name__ == '__main__':
    for sort_by_chance in [False, True]:
        data, sort = make_data(sort_by_chance)
        plot_joined(data, sort_by_chance=sort_by_chance)
    data, sort = make_data()
    write_latex(data)
    write_contrast_latex()
    plot_compare_methods(sort)
    # plot_compare_methods(sort, ablation='latent_size')
    # plot_compare_methods(sort, ablation='posthoc')
    # plot_compare_methods(sort, ablation='gm')
    # plot_compare_methods(sort, ablation='transfer')
    # plot_compare_methods(sort, ablation='dropout')
    # plot_compare_methods(sort, ablation='l2')
    # plot_gain_vs_size_multi()
    # plot_weight_power()
