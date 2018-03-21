# Baseline logistic
import json
import os
import re
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cogspaces.datasets.utils import get_output_dir


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
    #
    # pd.to_pickle(res, join(expanduser('~/output/cogspaces/all_pairs.pkl')))
    res = pd.read_pickle(join(expanduser('~/output/cogspaces/all_pairs.pkl')))

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
    diff = result['diff'].groupby(level=['target', 'help']).aggregate(
        {'mean': np.mean, 'std': np.std})
    print(diff)

    studies_list = res.index.get_level_values('target').unique().values
    n_studies = len(studies_list)

    scores = np.zeros((n_studies, n_studies, 2))
    for i in range(n_studies):
        for j in range(n_studies):
            scores[i, j, :] = diff.loc[(studies_list[i],
                                        studies_list[j])]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    caxes = []
    scores_max = scores[:, :, 0] + scores[:, :, 1]
    scores_min = scores[:, :, 0] - scores[:, :, 1]
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
    plt.subplots_adjust(top=0.95, bottom=0.35, left=0.1, right=0.98, wspace=0.05)

    plt.savefig(join(get_output_dir(), 'transfer.pdf'))


if __name__ == '__main__':
    summarize_all_pairs()
