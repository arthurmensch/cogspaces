import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, ticker
from os.path import expanduser, join

from cogspaces.datasets.utils import get_output_dir
from exps.grids.summarize import get_chance_subjects

idx = pd.IndexSlice


def plot_joined():
    output_dir = get_output_dir()

    factored_output_dir = join(output_dir, 'seed_split_init')
    baseline_output_dir = join(output_dir, 'reduced_logistic')

    factored = pd.read_pickle(join(factored_output_dir,
                                   'gathered.pkl'))
    baseline = pd.read_pickle(join(baseline_output_dir, 'gathered.pkl'))

    chance_level, n_subjects = get_chance_subjects()

    joined = pd.concat([factored, baseline, chance_level, n_subjects],
                       keys=['factored', 'baseline'], axis=1)
    joined['diff'] = joined['factored'] - joined['baseline']
    joined_mean = joined.groupby('study').aggregate(['mean', 'std'])
    joined_mean['chance'] = chance_level
    joined_mean['n_subjects'] = n_subjects

    data = pd.read_pickle(join(output_dir, 'joined_mean.pkl'))

    with open(expanduser('~/work/repos/cogspaces/cogspaces/'
                         'datasets/brainpedia.json')) as f:
        names = json.load(f)

    data = data.sort_values(('diff', 'mean'), ascending=True)

    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[2, 1]
                           )
    gs.update(top=1, bottom=0.05, left=0.26, right=0.98)
    fig = plt.figure(figsize=(10, 11))
    ax2 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax2)
    n_study = data.shape[0]

    ind = np.arange(n_study) * 2 + .5
    width = 1.2
    diff_color = plt.get_cmap('tab10').colors[2]
    baseline_color = plt.get_cmap('tab10').colors[0]
    transfer_color = plt.get_cmap('tab10').colors[1]
    rects = ax1.barh(ind, data[('diff', 'mean')], width,
                    color=diff_color, alpha=0.8)
    errorbar = ax1.errorbar(data[('diff', 'mean')], ind,
                            xerr=data[('diff', 'std')], elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=diff_color,
                            alpha=.5)
    ax1.set_xlabel('Accuracy gain')
    ax1.spines['left'].set_position('zero')
    plt.setp(ax1.get_yticklabels(), visible=False)

    ax1.set_xlim([-0.025, 0.2])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    ind = np.arange(n_study) * 2

    width = .8
    rects1 = ax2.barh(ind, data[('baseline', 'mean')], width,
                     color=baseline_color, alpha=.8)
    errorbar = ax2.errorbar(data[('baseline', 'mean')], ind,
                            xerr=data[('baseline', 'std')],
                            elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=baseline_color,
                            alpha=.5)
    rects2 = ax2.barh(ind + width, data[('factored', 'mean')], width,
                     color=transfer_color, alpha=.8)
    errorbar = ax2.errorbar(data[('factored', 'mean')], ind + width,
                            xerr=data[('factored', 'std')],
                            elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=transfer_color,
                            alpha=.5)

    lines = ax2.vlines([data['chance']],
                       ind - width / 2, ind + 3 * width / 2, colors='r',
                       linewidth=1, linestyles='--')

    ax2.set_ylim([-1, 2 * data.shape[0]])
    ax2.set_xlim([0., 0.95])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.set_xlabel('Decoding accuracy on test set')


    ax2.set_yticks(ind + width / 2)

    labels = ['(%s) %s' % (label, names[label]['title']) if label in names else label
              for label in data.index.values]
    ax2.set_yticklabels(labels, ha='right',
                        va='center')
    sns.despine(fig)

    plt.savefig(join(output_dir, 'joined_mean.pdf'))
    plt.show()
    plt.close(fig)

def plot_compare_methods():
    output_dir = get_output_dir()


    init_refit = pd.read_pickle(join(output_dir, 'init_refit/gathered.pkl'))
    logistic_refit = pd.read_pickle(join(output_dir, 'logistic_refit_l2/gathered.pkl'))

    single_factored = pd.read_pickle(join(output_dir, 'single_factored/gathered.pkl'))['score']
    seed_split_init = pd.read_pickle(join(output_dir, 'seed_split_init/gathered.pkl'))

    reduced_logistic = pd.read_pickle(join(output_dir, 'reduced_logistic/gathered.pkl'))

    dl_rest_init = init_refit[idx[:, 'dl_rest_init', :]]
    dl_rest_init_logistic = logistic_refit[idx[:, 'dl_rest_init', :]]

    reduced_logistic.name = 'score'
    single_factored.name = 'score'
    dl_rest_init.name = 'score'
    seed_split_init.name = 'score'
    dl_rest_init_logistic.name = 'score'

    df = pd.concat(
        [reduced_logistic, single_factored, dl_rest_init, dl_rest_init_logistic, seed_split_init],
        axis=0, keys=['Logistic', 'Factored single', 'Factored interpretable (logistic dropout)', 'Factored interpretable (logistic l2)',
                      'Factored'], names=['method'])
    print((df.loc['Factored interpretable (logistic dropout)'] - df.loc['Factored']).groupby('study').aggregate(['mean', 'std']))

    df_std = []
    methods = []
    for method, sub_df in df.groupby('method'):
        methods.append(method)
        df_std.append(sub_df.loc[method] - df.loc['Logistic'])
    df_std = pd.concat(df_std, keys=methods, names=['method'])

    mean = df_std.reset_index().groupby(
        by=['method', 'study']).mean().reset_index().sort_values(
        ['method', 'score'], ascending=False).set_index(['method', 'study'])

    sort = mean.loc['Factored'].index.get_level_values(level='study')
    sort = pd.MultiIndex.from_product([methods, sort],
                                      names=['method', 'study'])

    df_sort = df_std.reset_index('seed').loc[sort]

    df_sort = df_sort.reset_index()

    df_sort = df_sort.query("method != 'Logistic'")

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    params = dict(x="score", y="method", hue="study",
                  data=df_sort, dodge=True, ax=ax)
    g = sns.boxplot(zorder=100, showfliers=False, whis=0, **params)
    handles, labels = g.get_legend_handles_labels()
    g = sns.stripplot(alpha=1, zorder=200, size=3, linewidth=1, jitter=True, **params)
    ax.set_xlim([-0.1, 0.2])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel('Increase in accuracy compared to logistic baseline')
    ax.set_ylabel('')
    ax.spines['left'].set_position('zero')
    ax.set_yticklabels([])
    ax.set_yticks([])

    y_labels = ['Factored model \n (single study)',
                        'Factored model \n (multi-study, interpre- \n table, logistic dropout + BN)',
                        'Factored model \n (multi-study, interpre- \n table, logistic l2)',
                        'Factored model \n (multi-study, single \n train phase)']
    for i, label in enumerate(y_labels):
        ax.annotate(label, xy=(-0.1, i), xycoords='data', ha='right')
    sns.despine(fig)
    ax.legend(handles, labels, ncol=1, bbox_to_anchor=(1, 1),
              fontsize=7, loc='upper left', title='Study')
    fig.subplots_adjust(left=0.18, right=0.8, top=1, bottom=0.1)
    plt.savefig(join(output_dir, 'comparison_method.pdf'))
    # plt.show()
    plt.close(fig)


def plot_size_vs_transfer():
    pass


if __name__ == '__main__':
    # plot_joined()
    plot_compare_methods()