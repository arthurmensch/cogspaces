import json
import os
from json import JSONDecodeError
from math import sqrt
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cogspaces.pipeline import get_output_dir

basedir_ids = [52]
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
        datasets = config['datasets']
        datasets = '__'.join(datasets)
        weights = config['dataset_weights']
        seed = config['seed']
        score = info['score']
        rank = info['rank']
        res = {'datasets': datasets, 'seed': seed, 'rank': rank, }
        for dataset, weight in weights.items():
            res['%s_weight' % dataset] = weight
        for key, value in score.items():
            res[key] = value
        res_list.append(res)
res = pd.DataFrame(res_list)

df_agg = res.groupby(
    by=['archi_weight', 'hcp_weight', 'brainomics_weight']).aggregate(
    ['mean', 'std'])

df_agg = df_agg.fillna(0)

labels = ['archi', 'hcp', 'brainomics']
columns = {label: "test_%s" % label for label in labels}

transfer_df = df_agg[list(columns.values())]

C = np.array(transfer_df.index.tolist())

points = np.array([[0.5, -sqrt(3) / 2], [0, 1], [-0.5, -sqrt(3) / 2]])

coords = C.dot(points)

fix, axes = plt.subplots(1, 3, figsize=(9, 3))
for i, label in enumerate(labels):
    column = columns[label]
    ax = axes[i]
    sub_df = transfer_df[column]
    mean = sub_df['mean'].values
    print(label, mean.max())
    argmax = np.argmax(mean)
    weights = C[argmax]
    coord_max = coords[argmax]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=mean, cmap='viridis',
                     vmin=0.8, vmax=1)
    ax.plot([coord_max[0]], [coord_max[1]], marker='o',
            markersize=3, color='red')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label)
    for label, point in zip(labels, points):
        ax.annotate(label, xy=point, xycoords='data')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
plt.colorbar(sc)
plt.show()
