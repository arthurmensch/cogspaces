from os.path import join

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

baseline = pd.read_pickle(
    join(output_dir, 'logistic_tc', 'accuracies_mean.pkl'))
factored = pd.read_pickle(
    join(output_dir, 'training_curves', 'accuracies_mean.pkl'))

data = pd.concat((baseline, factored), keys=['baseline', 'factored'],
                 names=['method'])

subjects = get_full_subjects()

fig, axes = plt.subplots(2, 2)
axes = axes.ravel()

names = {'archi': "Pinel et al. '07", 'brainomics': "Papadopoulos-Orfanos '12",
         'camcan': "CamCan (Shafto et al. '14)",
         'henson2010faces': "Henson et al. '10"}

for i, (ax, (study, this_data)) in enumerate(
        zip(axes, data.groupby(level='study'))):
    ax.annotate(names[study], xy=(0.5, 1.), xytext=(0, 5),
                textcoords='offset points',
                xycoords='axes fraction', va='top',
                ha='center')
    handles = []
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
        if i in [2, 3]:
            ax.set_xlabel('Train size')
        if i == 1:
            l = ax.legend(handles, ['Standard decoding', 'Multi-study decoder'])
            l.set_zorder(20)
        ax.set_xticks(x)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
        ax.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=1, decimals=0))
        if i in [0, 2]:
            ax.set_ylabel('Test accuracy')
plt.subplots_adjust(left=0.12, right=0.98, top=0.98)
plt.savefig(join(output_dir, 'training_curves.pdf'))
