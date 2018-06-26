import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import load, Memory
from matplotlib import patches, gridspec
from matplotlib.patches import ConnectionPatch
from nilearn.input_data import NiftiMasker
from os.path import join, expanduser
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram

from cogspaces.datasets.utils import get_output_dir, fetch_mask

method = 'average'
os.chdir(join(get_output_dir(), 'figure_4', 'classifs'))


def plot_correlation_dendro(dist, Z, figure=None, ax=None,
                            left=True, top=True,
                            truncate_level=None):
    if figure is None:
        fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
    else:
        fig = figure

    if truncate_level is not None:
        dendro_params = dict(truncate_mode='level', p=truncate_level)
    else:
        dendro_params = {}

    n_rows = 1 + top
    n_cols = 1 + left
    height_ratios = [1, 3] if top else [1]
    width_ratios = [1, 3] if left else [1]
    sub_gs = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,
                                              subplot_spec=ax,
                                              wspace=0.01,
                                              width_ratios=width_ratios,
                                              height_ratios=height_ratios,
                                              hspace=0.01)
    if top and left:
        ax_empty = fig.add_subplot(sub_gs[0, 0])
        ax_empty.axis('off')

    if left:
        if top:
            ax_ldendro = fig.add_subplot(sub_gs[1, 0])
        else:
            ax_ldendro = fig.add_subplot(sub_gs[0, 0])

        dendrogram(Z, orientation='left', ax=ax_ldendro, **dendro_params)
        ax_ldendro.set_xticks([])
        ax_ldendro.set_yticks([])
        ax_ldendro.axis('off')
    else:
        ax_ldendro = None

    if top:
        if left:
            ax_tdendro = fig.add_subplot(sub_gs[0, 1])
        else:
            ax_tdendro = fig.add_subplot(sub_gs[0, 0])
        dendrogram(Z, orientation='top', ax=ax_tdendro, **dendro_params)
        ax_tdendro.set_xticks([])
        ax_tdendro.set_yticks([])
        ax_tdendro.axis('off')
    else:
        ax_tdendro = None

    # Plot distance matrix.
    if left and top:
        ax_matrix = fig.add_subplot(sub_gs[1, 1])
    elif left:
        ax_matrix = fig.add_subplot(sub_gs[0, 1])
    elif top:
        ax_matrix = fig.add_subplot(sub_gs[1, 0])
    else:
        ax_matrix = fig.add_subplot(sub_gs[0, 0])

    ax_matrix.matshow(dist, aspect='auto', origin='lower', vmax=1, vmin=-1)
    ax_matrix.set_xticks([])
    ax_matrix.set_yticks([])
    return ax_matrix, ax_ldendro, ax_tdendro


def cosine_dist(comp):
    S = np.sqrt(np.sum(comp ** 2, axis=1))
    x = 1 - (comp.dot(comp.T)
             / S[:, None]
             / S[None, :])
    return x


def euclidian_dist(classif):
    S = np.sum(classif ** 2, axis=1)
    G = classif.dot(classif.T)
    return np.sqrt(-2 * G + S[None, :] + S[:, None])


def get_dist(classif, distance='cosine'):
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    classif = masker.transform(classif)
    if distance == 'euclidean':
        return euclidian_dist(classif)
    elif distance == 'cosine':
        return cosine_dist(classif)


def get_linkage(dist):
    n = dist.shape[0]
    y = dist[np.triu_indices(n, 1)]
    Z = linkage(y, method=method, optimal_ordering=True)
    sort = dendrogram(Z, no_plot=True)
    sort = sort['leaves']
    c, _ = cophenet(Z, y)
    return Z, sort, c


mem = Memory(cachedir=expanduser('~/cache'))
width = 11
height = 7.35
scale = 15 / 11
fig = plt.figure(figsize=(width * scale, height * scale))
fig.subplots_adjust(left=0.0, top=1, bottom=.35 / 7.35, right=1 - 4 / 11)
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 3], height_ratios=[4, 3],
                       wspace=0.05, hspace=0.05)

names = load('names.pkl')
dist = mem.cache(get_dist)('classifs_factored.nii.gz')
Z, sort, c = get_linkage(dist)
dist = dist[sort][:, sort]
sorted_names = np.array(names)[sort]
corr = 1 - dist
mean_corr = np.mean(np.abs(corr))

ax_matrix_full, _, _ = plot_correlation_dendro(corr, Z, figure=fig,
                                               ax=gs[0, 0], truncate_level=6)

ax_matrix_full.annotate('Task-network decoder\n'
                        'Cophenetic: %.3f\n'
                        'Mean abs. corr.: %.3f' % (c, mean_corr),
                        bbox={'facecolor': 'black',
                              'boxstyle': 'round',
                              'linewidth': 0},
                        xy=(.3, .9), xycoords='axes fraction', ha='center',
                        va='center', fontsize=12,
                        color='white'
                        )

x1, x2 = 312, 352
# x1, x2 = 250, 290
# Zoom rectangle
rect = patches.Rectangle((x1 - .5, x1 - .5), x2 - x1, x2 - x1,
                         linewidth=2,
                         edgecolor='black', facecolor='none')
ax_matrix_full.add_patch(rect)

ax_matrix, ax_ldendro, ax_tdendro = plot_correlation_dendro(
    corr, Z, figure=fig, ax=gs[0, 1], left=False, truncate_level=None)
ax_matrix.set_xlim([x1 - .5, x2 - .5])
ax_matrix.set_ylim([x1 - .5, x2 - .5])
ax_tdendro.set_xlim([x1 * 10, x2 * 10])
ax_matrix.set_yticks(range(x1, x2))
labels = [label.replace('_', ' ').replace('&', ' ')
          for label in sorted_names[x1:x2]]
ax_matrix.set_yticklabels(labels)
ax_matrix.yaxis.tick_right()
ax_matrix.yaxis.set_label_position("right")

con = ConnectionPatch(xyA=(x2, x2), xyB=(0, 1),
                      coordsA="data", coordsB="axes fraction",
                      axesA=ax_matrix_full, axesB=ax_matrix, arrowstyle="-",
                      linewidth=1.5, antialiased=True,
                      linestyle=':')
ax_matrix_full.add_artist(con)
con = ConnectionPatch(xyA=(x2, x1), xyB=(0, 0),
                      coordsA="data", coordsB="axes fraction",
                      antialiased=True,
                      linewidth=1.5,
                      axesA=ax_matrix_full, axesB=ax_matrix, arrowstyle="-",
                      linestyle=':')
ax_matrix_full.add_artist(con)

# Annotations zoom
ax_matrix.vlines(x1 + np.array([3, 23, 27, 36]) - .5, x1 - .5, x2 - .5)
ax_matrix.hlines(x1 + np.array([3, 23, 27, 36]) - .5, x1 - .5, x2 - .5)
ax_matrix.annotate('Calculation',
                   bbox={'facecolor': 'black',
                         'boxstyle': 'round',
                         'linewidth': 0},
                   xy=(14 + x1, 6 + x1), xycoords='data', ha='center',
                   va='center', fontsize=12,
                   color='white')
ax_matrix.annotate('Stop',
                   bbox={'facecolor': 'black',
                         'boxstyle': 'round',
                         'linewidth': 0},
                   xy=(24.5 + x1, 21.5 + x1), xycoords='data', ha='center',
                   va='center', fontsize=12,
                   color='white')
ax_matrix.annotate('Left motor',
                   xy=(31 + x1, 34.5 + x1), xycoords='data', ha='center',
                   bbox={'facecolor': 'black',
                         'boxstyle': 'round',
                         'linewidth': 0},
                   color='white',
                   fontsize=12,
                   va='center')

###########################################################################
# Baseline
names = load('names.pkl')
dist = mem.cache(get_dist)('classifs_demean.nii.gz')
factored_sorted_dist = dist[sort][:, sort]
Z, sort, c = get_linkage(dist)

full_sorted_dist = dist[sort][:, sort]
full_sorted_corr = 1 - full_sorted_dist
factored_sorted_corr = 1 - factored_sorted_dist
mean_corr = np.mean(np.abs(full_sorted_corr))
full_sorted_names = np.array(names)[sort]
# x1, x2 = 250, 280


ax_matrix_baseline_full, ax_ldendro, _ = plot_correlation_dendro(
    full_sorted_corr, Z,
    figure=fig, top=False,
    left=True,
    ax=gs[1, 0],
    truncate_level=6)

ax_matrix_baseline_full.annotate('Voxel decoder\n'
                                 'C = %.3f\n'
                                 '$| \\bar c_{i, j} |$ = %.3f' % (
                                     c, mean_corr),
                                 bbox={'facecolor': 'black',
                                       'boxstyle': 'round',
                                       'linewidth': 0},
                                 xy=(.3, .9), xycoords='axes fraction',
                                 ha='center',
                                 va='center', fontsize=12,
                                 color='white'
                                 )

# rect = patches.Rectangle((x1 - .5, x1 - .5), x2 - x1, x2 - x1,
#                          linewidth=2,
#                          edgecolor='black', facecolor='none')
# ax_matrix.add_patch(rect)

ax_matrix_baseline, _, _ = plot_correlation_dendro(factored_sorted_corr, Z,
                                                   figure=fig, left=False,
                                                   top=False,
                                                   ax=gs[1, 1],
                                                   truncate_level=6)
# ax_tdendro.clear()
# ax_tdendro.axis('off')

ax_matrix_baseline.set_xlim([x1 - .5, x2 - .5])
ax_matrix_baseline.set_ylim([x1 - .5, x2 - .5])

ax_matrix_baseline.set_yticks(range(x1, x2))
ax_matrix_baseline.set_yticklabels(labels)
ax_matrix_baseline.yaxis.tick_right()
ax_matrix_baseline.yaxis.set_label_position("right")

# ax_matrix_baseline_full.set_axisbelow('line')
con = ConnectionPatch(xyA=(x2, x1), xyB=(0, 1),
                      coordsA="data", coordsB="axes fraction",
                      axesA=ax_matrix_full, axesB=ax_matrix_baseline,
                      linestyle=":",
                      antialiased=True, zorder=10,
                      linewidth=1.5)
ax_matrix_full.add_artist(con)
con = ConnectionPatch(xyA=(x1, x1), xyB=(0, 0),
                      coordsA="data", coordsB="axes fraction",
                      linewidth=1.5,
                      antialiased=True, zorder=10,
                      axesA=ax_matrix_full, axesB=ax_matrix_baseline,
                      linestyle=":")
ax_matrix_full.add_artist(con)
ax_matrix_full.set_zorder(1)
ax_matrix_baseline_full.set_zorder(-1)

ax_ldendro.annotate('Hierarchical\nclustering',
                    xy=(.5, -.1),
                    ha='center', fontsize=14,
                    xycoords='axes fraction')

ax_matrix_baseline_full.annotate('Correlation between classification maps\n'
                                 '(sorted using average linkage)',
                                 xy=(.5, -.1),
                                 ha='center', fontsize=14,
                                 xycoords='axes fraction'
                                 )
ax_matrix_baseline.annotate('Zoom on a subset of classification maps',
                            xy=(.5, -.1), fontsize=14,
                            ha='center',
                            xycoords='axes fraction'
                            )

fig.savefig('dendrogram.svg')
fig.savefig('dendrogram.pdf')
