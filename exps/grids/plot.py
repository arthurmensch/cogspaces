import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec, ticker
from os.path import expanduser, join


def plot_joined():
    output_dir = expanduser('~/output/cogspaces/')
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

    print(ind)
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

def plot_size_vs_transfer():


if __name__ == '__main__':
    plot_joined()