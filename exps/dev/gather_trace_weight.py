from copy import copy

import json
import os
from json import JSONDecodeError
from math import sqrt
from os.path import join

from matplotlib import colors

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from cogspaces.pipeline import get_output_dir

basedir_ids = [53]
basedirs = [join(get_output_dir(), 'predict_multi', str(_id), 'run') for _id in
            basedir_ids]
res_list = []
for basedir in basedirs:
    for exp_dir in os.listdir(basedir):
        exp_dir = join(basedir, exp_dir)
        try:
            config = json.load(open(join(exp_dir, 'config.json'), 'r'))
            info = json.load(open(join(exp_dir, 'info.json'), 'r'))
        except (JSONDecodeError, FileNotFoundError):
            continue
        cat_datasets = config['datasets']
        cat_datasets = '__'.join(cat_datasets)
        weights = config['dataset_weights']
        seed = config['seed']
        score = info['score']
        rank = info['rank']
        res = {'datasets': cat_datasets, 'seed': seed, 'rank': rank}
        for dataset, weight in weights.items():
            res['%s_weight' % dataset] = weight
        for key, value in score.items():
            res[key] = value
        res_list.append(res)
res = pd.DataFrame(res_list)
print(len(res))

df_agg = res.groupby(
    by=['archi_weight', 'hcp_weight', 'brainomics_weight']).aggregate(
    ['mean', 'std'])
df_agg = df_agg.fillna(0)

datasets = ['archi', 'hcp', 'brainomics', 'mean']
ann_datasets = ['archi', 'hcp', 'brainomics']
columns = {label: "test_%s" % label for label in datasets}

transfer_df = df_agg[list(columns.values())]

C = np.array(transfer_df.index.tolist())

points = np.array([[0.5, -sqrt(3) / 2], [0, 1], [-0.5, -sqrt(3) / 2]])
offsets = np.array([[10, 0], [0, 10], [-30, 0]])
coords = C.dot(points)

fig, axes = plt.subplots(1, len(datasets), figsize=(len(datasets) * 3.5, 3))
fig.subplots_adjust(top=0.8)

palette = copy(plt.cm.viridis)
# palette.set_under('k', 1.0)
for i, dataset in enumerate(datasets):
    column = columns[dataset]
    ax = axes[i]
    sub_df = transfer_df[column]
    mean = sub_df['mean'].values
    print(dataset, mean.max())
    argmax = np.argmax(mean)
    weights = C[argmax]
    coord_max = coords[argmax]
    ax.plot(points[0:2, 0], points[0:2, 1], 'k-', lw=1, zorder=0)
    ax.plot(points[1:3, 0], points[1:3, 1], 'k-', lw=1, zorder=0)
    ax.plot(points[[2, 0], 0], points[[2, 0], 1], 'k-', lw=1, zorder=0)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=mean, cmap=palette,
                    norm=colors.Normalize(vmin=mean.max() - 0.03,
                                          vmax=mean.max()),
                    zorder=1
                    )
    ax.plot([coord_max[0]], [coord_max[1]], marker='o',
            markersize=3, color='red', zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(dataset)
    for ann_dataset, point, offset in zip(ann_datasets, points, offsets):
        ax.annotate(ann_dataset, xy=point, xycoords='utils',
                    textcoords='offset points', xytext=offset)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    plt.colorbar(sc, ax=ax, extend='min', shrink=.9)
fig.suptitle('Test accuracy varying dataset weights')
analysis_dir = join(get_output_dir(), 'predict_multi', str(53), 'analysis')
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)
plt.savefig(join(analysis_dir, 'analysis.pdf'))
for dataset in ann_datasets:
    df_agg.sort_index(level='%s_weight' % dataset, inplace=True)
    df_agg.to_html(join(analysis_dir, 'result_%s.html' % dataset))
