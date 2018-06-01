# Baseline logistic
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import gridspec, ticker
from os.path import expanduser, join

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir
from exps.analyse.maps import introspect

idx = pd.IndexSlice


def summarize():
    # output_dir = [expanduser('~/output/cogspaces/weight_power'), ]
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
    #         test_scores = run['result']
    #         if test_scores is None:
    #             print('Skipping exp %i' % this_dir)
    #             continue
    #         seed = config['seed']
    #         weight_power = config['factored_variational']['weight_power']
    #         this_res = dict(seed=seed, weight_power=weight_power,
    #                         **test_scores)
    #         res.append(this_res)
    #
    # res = pd.DataFrame(res)
    # res = res.set_index(['weight_power', 'seed'])
    # pd.to_pickle(res,
    #              join(expanduser(
    #                  '~/output/cogspaces/weight_power.pkl')))
    res = pd.read_pickle(join(expanduser(
        '~/output/cogspaces/weight_power.pkl')))
    baseline = pd.read_pickle(
        join(expanduser('~/output/cogspaces/baseline_seed.pkl')))['test_score']
    for weight_power, this_res in res.groupby('weight_power'):
        # Transpose
        studies = this_res.columns
        this_res = [this_res[study] for study in studies]
        this_res = pd.concat(this_res, keys=studies, names=['study'])
        this_res = this_res.reset_index('weight_power', drop=True)
        pd.to_pickle(this_res,
                     join(expanduser(
                         '~/output/cogspaces/weight_power_seed_%s.pkl' % weight_power)))
        this_res = pd.concat((this_res, baseline), axis=1,
                             keys=['score_variational', 'score_baseline'],
                             join='inner')
        this_res['diff'] = this_res['score_variational'] - this_res[
            'score_baseline']
        this_res = this_res.groupby('study').aggregate(['mean', 'std'])
        pd.to_pickle(this_res,
                     (expanduser(
                         '~/output/cogspaces/joined_weight_power_%s.pkl' % weight_power)))


def plot(weight_power):
    _, target = load_data_from_dir(
        data_dir=join(get_data_dir(), 'reduced_512'))

    chance_level = {}
    n_subjects = {}
    for study, this_target in target.items():
        chance_level[study] = 1. / len(this_target['contrast'].unique())
        n_subjects[study] = len(this_target['subject'].unique())

    output_dir = expanduser('~/output/cogspaces/')
    data = pd.read_pickle(join(output_dir, 'joined_weight_power_%s.pkl' % weight_power))
    data = data.sort_values(('diff', 'mean'), ascending=False)
    gs = gridspec.GridSpec(2, 2,
                           height_ratios=[1, 2],
                           width_ratios=[3, 1]
                           )
    gs.update(top=0.98, bottom=0.3, left=0.1, right=0.98)
    fig = plt.figure(figsize=(11, 6))
    ax2 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)
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
                            alpha=.3)
    ax1.set_ylabel('Accuracy gain')
    ax1.spines['bottom'].set_position('zero')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.set_ylim([-0.025, 0.2])
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    ind = np.arange(n_study) * 2

    width = .8
    rects1 = ax2.bar(ind, data[('score_baseline', 'mean')], width,
                     color=baseline_color, alpha=.8)
    errorbar = ax2.errorbar(ind, data[('score_baseline', 'mean')],
                            yerr=data[('score_baseline', 'std')],
                            elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=baseline_color,
                            alpha=.3)
    rects2 = ax2.bar(ind + width, data[('score_variational', 'mean')], width,
                     color=transfer_color, alpha=.8)
    errorbar = ax2.errorbar(ind + width, data[('score_variational', 'mean')],
                            yerr=data[('score_variational', 'std')],
                            elinewidth=1.5,
                            capsize=2, linewidth=0, ecolor=transfer_color,
                            alpha=.3)

    lines = ax2.hlines([chance_level[study] for study in data.index.values],
                       ind - width / 2, ind + 3 * width / 2, colors='r',
                       linewidth=1, linestyles='--')

    ax2.set_ylim([0., 0.95])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.set_ylabel('Decoding accuracy on test set')

    ax2.set_xticks(ind + width / 2)
    ax2.set_xticklabels(data.index.values, rotation=60, ha='right',
                        va='top')
    # plt.savefig(join(output_dir, 'comparison_variational_sym_2.pdf'))
    # plt.close(fig)
    # plt.show()

    ax3 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3.spines['bottom'].set_position('zero')

    n_subjects = [n_subjects[study] for study in data.index.values]
    mean = data[('diff', 'mean')]
    std = data[('diff', 'std')]
    ax3.scatter(n_subjects, mean, color=diff_color, s=20, zorder=100)
    ax3.errorbar(n_subjects, mean, yerr=std, linewidth=0, elinewidth=1.5,
                 capsize=2, ecolor=diff_color, alpha=0.3, zorder=101)
    ax3.set_xlabel('# subjects in dataset')
    ax3.set_ylim([-0.025, 0.2])
    ax3.set_xscale('log')
    ax3.set_xticks([8, 16, 32, 100, 500])
    ax3.set_xticklabels([8, 16, 32, 100, 500])

    # ax3.set_ylim([-0.025, 0.2])
    # ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    # ax3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    ax3.legend((rects1[0], rects2[0], rects[0], lines),
               ('Single study decoding', 'Multi-study decoding',
                'Accuracy gain', 'Chance level'), bbox_to_anchor=(0.5, -.5),
               loc='upper center')

    sns.despine(fig)

    plt.savefig(join(output_dir, 'weight_power_%s.pdf' % weight_power))
    plt.close(fig)


def map():
    output_dir = [expanduser('~/output/cogspaces/variational_sym_2'), ]

    regex = re.compile(r'[0-9]+$')
    exp_dirs = []
    for this_output_dir in output_dir:
        for this_dir in filter(regex.match, os.listdir(this_output_dir)):
            this_exp_dir = join(this_output_dir, this_dir)
            if os.path.exists(join(this_exp_dir, 'estimator.pkl')):
                exp_dirs.append(this_exp_dir)
    print(exp_dirs)
    Parallel(n_jobs=1, verbose=10)(delayed(introspect)(exp_dir)
                                   for exp_dir in exp_dirs)


#

if __name__ == '__main__':
    # map_variational()
    # summarize_baseline()
    # summarize()
    for weight_power in ['0.0', '0.25', '0.5', '0.71', '1.0']:
        plot(weight_power)
    # summarize_factored    ()
    # summarize_study_selection()
    # plot_study_selection()
    # plot()
