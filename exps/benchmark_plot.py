import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.cm import get_cmap
import numpy as np

mpl.use('pdf')

import matplotlib.pyplot as plt

from os.path import expanduser

import pandas as pd

import seaborn as sns

df = pd.read_csv(expanduser('results.csv'), index_col=list(range(5)))

df = df.query("source == 'hcp_rs_positive_single'")

fig = plt.figure(figsize=(5.5015, 1.5))
#make outer gridspec
outer = gridspec.GridSpec(1, 2, hspace=.1)
#make nested gridspecs
gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                       hspace=.0)
gs2 = gridspec.GridSpecFromSubplotSpec(1, 2,
                                       subplot_spec=outer[1], hspace=.0)
axes = []
for i in range(2):
    axes.append(fig.add_subplot(gs1[i], zorder=100 if i == 0 else 0))
for i in range(2):
    axes.append(fig.add_subplot(gs2[i]))

fig.subplots_adjust(bottom=.1, left=.1, right=1, top=.8)

limits = {'archi': [0.75, 0.93],
          'brainomics': [0.75, 0.93],
          'camcan': [0.5, 0.68],
          'la5c': [0.5, 0.68]}
global_limits = [.52, .93]
keys = [3, 4, 2, 1, 0]
labels = ['Multinomial + dropout',
          'Multinomial L2',
          'Factored model + dropout',
          'Transfer from HCP',
          'Transfer from all datasets']
names = {'archi': 'Archi',
         'brainomics': 'Brainomics',
         'camcan': 'CamCan',
         'la5c': 'LA5C',
         }

colors = get_cmap('tab10').colors[:5]
bars = []
for ax, dataset in zip(axes, ['archi', 'brainomics', 'camcan', 'la5c']):
    dataset_df = df.loc[dataset]
    dataset_df.reset_index(['source', 'with_std'], drop=True,
                           inplace=True)
    data = dataset_df.iloc[keys]
    values = data['mean']
    stds = data['std']
    xs = [.5, 1.5, 2.5, 3.5, 4.5]
    these_bars = ax.bar(xs, values, width=.8, color=colors, label=[])
    bars.append(these_bars)
    for x, color, value, std in zip(xs, colors, values, stds):
        ax.errorbar(x, value, yerr=std, elinewidth=1.2,
                    capsize=2, linewidth=0, ecolor=color, alpha=.5)

    ax.annotate(names[dataset],
                xytext=(0, -3),
                textcoords='offset points',
                xy=(0.5, 0), va='top', ha='center', rotation=0,
                xycoords='axes fraction')
    # sns.despine(fig, ax)
    ax.set_ylim(limits[dataset])
    if dataset in ['brainomics', 'la5c']:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    else:
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:2.0f}\\%'.format(x * 100) for x in vals])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])

axes[0].set_ylabel('Test accuracy')

axes[0].legend(bars[0], labels, frameon=False, loc='lower left', ncol=3,
               bbox_to_anchor=(-.3, .93))

fig.savefig('ablation.pdf')




# res = res.query("Method in ['Transfer from HCP', 'Latent factors + dropout']")
# res = res.reset_index()
# idx = res.groupby(by=['Dataset', 'Method']).aggregate('idxmax')['mean'].values
# res = res.loc[idx]
# flatui = get_cmap('tab20').colors[1::2]
# x = []
# width = 0.75
# center = {'Archi': 1, 'Brainomics': 5, 'CamCan': 9, 'UCLA': 13}
# offset = {'Simple multinomial': - 2.5 * width,
#           'Dictionary projection': -1.5 * width,
#           'Multi-scale dictionary': - .5 * width,
#           'Latent factors + dropout': width / 2,
#           'Transfer from HCP': + 1.5 * width}
# colors = {'Simple multinomial': flatui[0],
#           'Dictionary projection': flatui[1],
#           'Multi-scale dictionary': flatui[2],
#           'Latent factors + dropout': flatui[3],
#           'Transfer from HCP': flatui[4]}
#
# method_label = {'Simple multinomial': 'Full multinomial',
#                 'Dictionary projection': 'Spatial projection',
#                 'Multi-scale dictionary': 'Multi-scale spatial projection',
#                 'Latent factors + dropout': 'Latent cognitive space (single study)',
#                 'Transfer from HCP': 'Latent cognitive space (multi-study)'}
#
# fig, ax = plt.subplots(1, 1, figsize=(5.5015, 1.3))
# fig.subplots_adjust(bottom=.15, left=.08, right=0.98, top=.98)
# for method in ['Simple multinomial', 'Dictionary projection',
#                'Multi-scale dictionary',
#                'Latent factors + dropout',
#                'Transfer from HCP']:
#     sub_res = res.query("Method == '%s'" % method)
#     y = sub_res['mean']
#     std = sub_res['std']
#     datasets = sub_res['Dataset']
#     x = []
#     for dataset in datasets:
#         x.append(center[dataset] + offset[method])
#         ax.bar(x, y, width=width, color=colors[method], yerr=std,
#                label=method_label[method])
#     for this_x, this_y in zip(x, y):
#         ax.annotate("%.1f" % (this_y * 100), xy=(this_x, this_y),
#                     xytext=(0, -8),
#                     textcoords='offset points', ha='center', va='center',
#                     xycoords='data')
#
# ax.set_xticks(np.array(list(center.values())) - width / 2)
# labels = list(center.keys())
# labels[-1] = 'LA5c'
# ax.set_xticklabels(labels)
# ax.set_ylabel('Test accuracy')
# ax.set_ylim(0.44, 0.91)
# ax.set_xlim(-1.25, 14.5)
# h, l = ax.get_legend_handles_labels()
# h_1 = [h[0]] + h[2:]
# l_1 = [l[0]] + l[2:]
# legend_1 = ax.legend(h_1, l_1, loc='center left', ncol=1,
#                      bbox_to_anchor=(.49, 0.74),
#                      columnspacing=0,
#                      frameon=False)
# ax.legend([h[1]], [l[1]], loc='center left', ncol=1,
#           bbox_to_anchor=(.75, 0.97),
#           columnspacing=0,
#           frameon=False)
# ax.add_artist(legend_1)
#
# ax.annotate("Err bars $1\,std$", xy=(0, 0))
# sns.despine(ax=ax)
#
# plt.savefig(expanduser('~/work/nips/scripts/ablation.pdf'))
