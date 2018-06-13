import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, ticker
from os.path import expanduser, join

from cogspaces.datasets.utils import get_output_dir
from exps.grids.gather_quantitative import get_chance_subjects

idx = pd.IndexSlice


def plot_joined():
    output_dir = get_output_dir()

    factored_output_dir = join(output_dir, 'factored_refit_cautious')
    baseline_output_dir = join(output_dir, 'logistic')

    factored = pd.read_pickle(join(factored_output_dir, 'gathered.pkl'))
    factored = factored[idx[:, 'dl_rest', :]]

    baseline = pd.read_pickle(join(baseline_output_dir, 'gathered.pkl'))['score']

    chance_level, n_subjects = get_chance_subjects()

    joined = pd.concat([factored, baseline, chance_level, n_subjects],
                       keys=['factored', 'baseline'], axis=1)
    joined['diff'] = joined['factored'] - joined['baseline']
    joined_mean = joined.groupby('study').aggregate(['mean', 'std'])
    joined_mean['chance'] = chance_level
    joined_mean['n_subjects'] = n_subjects

    joined_mean.to_pickle(join(output_dir, 'joined_mean.pkl'))
    data = pd.read_pickle(join(output_dir, 'joined_mean.pkl'))

    with open(expanduser('~/work/repos/cogspaces/cogspaces/'
                         'datasets/brainpedia.json')) as f:
        names = json.load(f)

    data = data.sort_values(('diff', 'mean'), ascending=True)

    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[2, 1]
                           )
    gs.update(top=0.98, bottom=0.06, left=0.23, right=0.98)
    fig = plt.figure(figsize=(7, 8))
    ax2 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax2)
    n_study = data.shape[0]

    ind = np.arange(n_study) * 2 + .5
    width = 1.2
    diff_color = sns.color_palette("husl", 35)
    diff_color_err = [(max(0, r - .1), max(0, g - .1), max(0, b - .1))
                      for r, g, b in diff_color]


    baseline_color = '0.75'
    baseline_color_err = '0.65'
    transfer_color = '0.1'
    transfer_color_err = '0.'
    rects = ax1.barh(ind, data[('diff', 'mean')], width,
                     color=diff_color)
    for this_x, this_y, this_xerr, this_color in zip(data[('diff', 'mean')],
                                                     ind,
                                                     data[('diff', 'std')],
                                                     diff_color_err):
        ax1.errorbar(this_x, this_y, xerr=this_xerr, elinewidth=1.5,
                     capsize=2, linewidth=0, ecolor=this_color,
                     alpha=.5)
    # errorbar = ax1.errorbar(data[('diff', 'mean')], ind,
    #
    #                         xerr=data[('diff', 'std')], elinewidth=1.5,
    #                         capsize=2, linewidth=0, ecolor=diff_color,
    #                         alpha=.5)
    ax1.set_xlabel('Accuracy gain')
    ax1.spines['left'].set_position('zero')
    plt.setp(ax1.get_yticklabels(), visible=False)

    ax1.set_xlim([-0.025, 0.17])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    ind = np.arange(n_study) * 2

    width = .8
    rects1 = ax2.barh(ind, data[('baseline', 'mean')], width,
                      color=baseline_color)
    errorbar = ax2.errorbar(data[('baseline', 'mean')], ind,
                            xerr=data[('baseline', 'std')],
                            elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=baseline_color_err,
                            alpha=.5)
    rects2 = ax2.barh(ind + width, data[('factored', 'mean')], width,
                      color=transfer_color)
    errorbar = ax2.errorbar(data[('factored', 'mean')], ind + width,
                            xerr=data[('factored', 'std')],
                            elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=transfer_color_err,
                            alpha=.5)

    lines = ax2.vlines([data['chance']],
                       ind - width / 2, ind + 3 * width / 2, colors='r',
                       linewidth=1, linestyles='--')

    ax2.set_ylim([-1, 2 * data.shape[0]])
    plt.setp(ax2.yaxis.get_ticklabels(), fontsize=8)
    ax2.set_xlim([0., 0.95])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.set_xlabel('Decoding accuracy on test set')

    handles = [rects1, rects2, lines]
    labels = ['Baseline decoder', 'Multi-study decoder', 'Chance level']
    ax2.legend(handles, labels, loc='lower right', frameon=False, bbox_to_anchor=(1.15, -.01))

    ax2.set_yticks(ind + width / 2)

    labels = [
        '%s' % (names[label]['title']) if False else label  # if label in names else label
        for label in data.index.values]
    ax2.set_yticklabels(labels, ha='right',
                        va='center')
    sns.despine(fig)

    plt.savefig(join(output_dir, 'joined_mean.pdf'))
    # plt.show()
    plt.close(fig)

    sort = data.index.values.tolist()[::-1]
    return sort


def plot_compare_methods(sort):
    output_dir = get_output_dir()

    factored_refit = pd.read_pickle(
        join(output_dir, 'factored_refit_cautious/gathered.pkl'))

    factored_selector = pd.read_pickle(
        join(output_dir, 'factored_study_selector/gathered.pkl'))

    single_factored = \
        pd.read_pickle(join(output_dir, 'single_factored/gathered.pkl'))['score']
    logistic = pd.read_pickle(
        join(output_dir, 'logistic/gathered.pkl'))['score']

    factored_refit = factored_refit[idx[:, 'dl_rest', :]]
    # factored_selector = factored_refit[idx[:, False, :]]

    logistic.name = 'score'
    single_factored.name = 'score'
    factored_refit.name = 'score'

    df = pd.concat(
        [logistic,
         single_factored,
         factored_refit,
         factored_selector
         ],
        axis=0, keys=['logistic',
                      'single_factored',
                      'factored_refit',
                      'factored_selector',
                      ], names=['method'])

    df_std = []
    methods = []
    for method, sub_df in df.groupby('method'):
        methods.append(method)
        df_std.append(
            sub_df.loc[method] - df.loc['logistic'].groupby('study').transform(
                'median'))
    df_std = pd.concat(df_std, keys=methods, names=['method'])

    median = df_std.reset_index().groupby(
        by=['method', 'study']).median().reset_index().sort_values(
        ['method', 'score'], ascending=False).set_index(['method', 'study'])

    sort = pd.MultiIndex.from_product([methods, sort],
                                      names=['method', 'study'])
    df_sort = df_std.reset_index('seed').loc[sort]

    df_sort = df_sort.reset_index()

    n_studies = len(df_sort['study'].unique())

    diff_color = sns.color_palette("husl", n_studies)

    print(df.groupby(['method', 'study']).aggregate(['mean', 'std']))

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    params = dict(x="score", y="method", hue="study",
                  data=df_sort, dodge=True, ax=ax,
                  palette=diff_color
                  )
    g = sns.stripplot(alpha=1, zorder=200, size=3.5, linewidth=0, jitter=True,
                      **params)
    handles, labels = g.get_legend_handles_labels()
    g = sns.boxplot(x="score", y="method", color='0.75', showfliers=False,
                    whis=1.5,
                    data=df_sort, ax=ax, zorder=100)
    ax.set_xlim([-0.1, 0.16])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel('Increase in accuracy compared to logistic median per study')
    ax.set_ylabel('')
    ax.spines['left'].set_position('zero')
    plt.setp(ax.spines['left'], zorder=2)
    ax.set_yticklabels([])
    ax.set_yticks([])

    methods = df_sort['method'].unique()

    y_labels = {'factored_refit': 'Multi-study \n decoder',
                'single_factored': 'Single-study \n decoder',
                'factored_selector': 'Multi-study \n decoder \n (non-universal)',
                'logistic': 'Baseline \n decoder'}
    for i, method in enumerate(methods):
        ax.annotate(y_labels[method], xy=(-0.1, i), xycoords='data',
                    va='center',
                    ha='right')
    sns.despine(fig)

    with open(expanduser('~/work/repos/cogspaces/cogspaces/'
                         'datasets/brainpedia.json')) as f:
        names = json.load(f)

    l = ax.legend(handles, [names[label]['title'] for label in labels], ncol=1,
                  bbox_to_anchor=(1.05, 1),
                  fontsize=4.5, loc='upper left')
    plt.setp(l, visible=False)
    fig.subplots_adjust(left=0.16, right=0.98, top=1, bottom=0.18)
    plt.savefig(join(output_dir, 'comparison_method.pdf'))
    # plt.show()
    plt.close(fig)


def plot_size_vs_transfer():
    pass


if __name__ == '__main__':
    sort = plot_joined()
    plot_compare_methods(sort)
