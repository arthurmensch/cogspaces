import matplotlib as mpl

import numpy as np
mpl.use('pdf')

from matplotlib import gridspec
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

import json
from os.path import join
import os

import pandas as pd

from cogspaces.pipeline import get_output_dir

from json import JSONDecodeError

output_dir = join(get_output_dir(), 'benchmark_trainsize')


def summarize():
    # 1, 4 hcp_rs_positive_single
    basedir_ids = [1, 4]
    # 5 hcp_new_big
    basedirs = [join(output_dir, str(_id), 'run')
                for _id in basedir_ids]
    res_list = []
    dataset_sizes = {'archi': 78, 'brainomics': 94, 'camcan': 605}
    for basedir in basedirs:
        for exp_dir in os.listdir(basedir):
            exp_dir = join(basedir, exp_dir)
            try:
                config = json.load(open(join(exp_dir, 'config.json'), 'r'))
                info = json.load(open(join(exp_dir, 'info.json'), 'r'))
            except (JSONDecodeError, FileNotFoundError):
                continue
            datasets = config['datasets']
            dataset = datasets[0]
            if len(datasets) > 1:
                helper_datasets = '__'.join(datasets[1:])
            else:
                helper_datasets = 'none'
            config['dataset'] = dataset
            train_size = config['train_size'][dataset]
            if train_size <= 1:
                train_size *= dataset_sizes[dataset]
            config['train_size'] = int(train_size)
            config['helper_datasets'] = helper_datasets
            score = info.pop('score')
            res = dict(**config, **info)
            for key, value in score.items():
                res[key] = value
            res_list.append(res)
    res = pd.DataFrame(res_list)

    df_agg = res.groupby(by=['dataset', 'helper_datasets',
                             'train_size']).aggregate(['mean', 'std', 'count'])

    df_agg = df_agg.fillna(0)

    results = {}
    for dataset in ['archi', 'brainomics', 'camcan']:
        results[dataset] = df_agg.loc[dataset]['test_%s' % dataset]

    results = pd.concat(results, names=['dataset'])
    results.to_csv(join(output_dir, 'results.csv'))
    results.to_pickle(join(output_dir, 'results.pkl'))
    print(results)


def plot():
    results = pd.read_csv(join(output_dir,
                               'results.csv'), index_col=[0, 1])
    print(results)
    fig = plt.figure(figsize=(5.5015, 1.2))
    fig.subplots_adjust(top=.85, bottom=.18, right=.95, left=.08)
    gs = gridspec.GridSpec(1, 3, hspace=.1)
    colors = get_cmap('tab10').colors[3:6]
    labels = ['No transfer', 'Transfer from HCP', 'Transfer from all datasets']
    datasets_names = ['Archi', 'Brainomics', 'Camcan']
    for i, (dataset, sub_res) in enumerate(results.groupby(level='dataset')):
        ax = fig.add_subplot(gs[i])
        handles = []
        sub_res = sub_res.sort_index(ascending=False)
        transfers = sub_res.index.unique()
        for j, transfer in enumerate(transfers):
            sub_sub_res = sub_res.loc[transfer]
            x = sub_sub_res['train_size']
            y = sub_sub_res['mean']
            y_err = sub_sub_res['std']
            ax.fill_between(x, y - y_err, y + y_err, color=colors[j],
                            alpha=.1)
            handle = ax.plot(x, y, color=colors[j], label=labels[j])
            handles.append(handle)
        if dataset == 'camcan':
            ax.set_xticks([5, 60, 100, 200, 300])
            ax.set_ylim([.4, .7])
            ax.set_yticks([.4, .5, .6, .7])
        else:
            if dataset == 'brainomics':
                ax.set_xticks([5, 10, 20, 30, 40, 50])
                ax.set_yticks([.6, .7, .8, .9])
            else:
                ax.set_xticks([5, 10, 20, 30, 40])
                ax.set_yticks([.65, .7, .8, .9])

        ax.annotate(datasets_names[i], xy=(.5, .1),
                    xycoords='axes fraction',
                    va='bottom', ha='center')
        if i == 0:
            ax.set_ylabel('Test accuracy')
            ax.set_xlabel('Train size')
            ax.annotate('Train \n subjects', xy=(0, 0),
                        xytext=(-30, 0),
                        textcoords='offset points',
                        xycoords='axes fraction',
                        va='top', ha='left')
            ax.legend(frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0.2, .93))
        ax.tick_params(axis='y', which='both', labelsize=6)
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:2.0f}\\%'.format(x * 100) for x in vals])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.savefig(join(output_dir, 'trainsize.pdf'))
    return fig

# summarize()
plot()