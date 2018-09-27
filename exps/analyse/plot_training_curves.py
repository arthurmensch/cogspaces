from os.path import join, expanduser

import matplotlib as mpl

mpl.rcParams['font.family'] = 'CMU Sans Serif'
mpl.rcParams['font.size'] = 13

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

from cogspaces.datasets.utils import get_output_dir
from exps.grids.gather_quantitative import get_full_subjects

output_dir = get_output_dir()

baselines = [
    pd.read_pickle(join(output_dir, 'logistic_tc', 'accuracies_mean.pkl')),
    pd.read_pickle(join(output_dir, 'logistic_tc_2', 'accuracies_mean.pkl'))]
baseline = pd.concat(baselines, axis=0)
factored = [
    pd.read_pickle(join(output_dir, 'training_curves', 'accuracies_mean.pkl')),
    pd.read_pickle(
        join(output_dir, 'training_curves_2', 'accuracies_mean.pkl'))]
factored = pd.concat(factored, axis=0)

data = pd.concat((baseline, factored), keys=['baseline', 'factored'],
                 names=['method'])

data = data.drop('henson2010faces', axis=0, level='study')

subjects = get_full_subjects()
print(subjects)

fig, axes = plt.subplots(1, 4, figsize=(10, 2))
axes = axes.ravel()

names = {'archi': "Pinel et al. '07", 'brainomics': "Papadopoulos-\nOrfanos '12",
         'camcan': "CamCan AV\n(Shafto et al. '14)",
         'ds009': "Cohen '09",
         'henson2010faces': "Henson et al. '10"}

for i, (ax, (study, this_data)) in enumerate(
        zip(axes, data.groupby(level='study'))):
    ax.annotate(names[study], xy=(0.5, 0.04), xytext=(0, 5),
                textcoords='offset points',
                xycoords='axes fraction', va='bottom',
                ha='center')
    handles = []
    ticks = {'archi': 0.05, 'brainomics': 0.1, 'camcan': 0.05, 'ds009': 0.2}
    for method, this_data_method in this_data.groupby(level='method'):
        y = this_data_method['mean']
        std = this_data_method['std']
        x = this_data_method.index.get_level_values('train_size')
        x *= subjects[study]
        x = np.floor(x)
        lines, = ax.plot(x, y, zorder=10, marker='.')
        handles.append(lines)
        ax.fill_between(x, y + std, y - std, alpha=0.5, zorder=5)
        sns.despine(fig, ax)
        if i == 0:
            ax.set_xlabel('                                '
                          '# training subjects in target study')
        if i == 2:
            l = ax.legend(handles,
                          ['Standard decoding', 'Multi-study decoder'], bbox_to_anchor=(0, -0.1), loc='upper left', ncol=2, frameon=False)
            l.set_zorder(20)
        ax.set_xticks(x)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[study]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(ticks[study] / 2))
        ax.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=1, decimals=0))
        plt.setp(ax.yaxis.get_ticklabels(), fontsize=10)
        if i == 0:
            ax.set_ylabel('Test accuracy')
plt.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.27, wspace=0.18)
plt.savefig(
    expanduser('~/work/papers/papers/nature/figures/training_curves.svg'))
