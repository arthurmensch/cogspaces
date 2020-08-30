from os.path import join

import matplotlib as mpl

from cogspaces.datasets.derivative import get_study_info

mpl.use('pgf')
mplparams = {
    "font.family": "sans-serif",
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
}
mpl.rcParams.update(mplparams)

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

import os

from cogspaces.datasets.utils import get_output_dir

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

info = get_study_info().groupby(by='study').first()
names = info['name_study'].map(lambda x: x.replace('&', '\&'))
subjects = info['#subjects']

fig, axes = plt.subplots(1, 4, figsize=(9, 1.6))
axes = axes.ravel()

offsets = {'archi': 0, 'brainomics': 0.05, 'camcan': .1, 'ds009': .2}

for i, (ax, (study, this_data)) in enumerate(
        zip(axes, data.groupby(level='study'))):
    ax.annotate(names[study], xy=(0.5 + offsets[study], 0.04), xytext=(0, 5),
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
            ax.set_xlabel('\# training subjects in target study')
            ax.xaxis.set_label_coords(1, -0.24)
        if i == 2:
            l = ax.legend(handles,
                          ['Single-study voxel-level decoding', 'Multi-study decoder'], bbox_to_anchor=(-.2, -0.12), loc='upper left', ncol=2, frameon=False)
            l.set_zorder(20)
        if study == 'brainomics':
            ax.set_ylim([.71, .96])
        ax.set_xticks(x)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[study]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(ticks[study] / 2))
        ax.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=1, decimals=0))
        plt.setp(ax.yaxis.get_ticklabels(), fontsize=10)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        if i == 0:
            ax.set_ylabel('Test accuracy')
plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.27, wspace=0.18)


save_dir = join(get_output_dir(output_dir=None), 'revision_output')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.savefig(join(save_dir, 'training_curves.pdf'))
