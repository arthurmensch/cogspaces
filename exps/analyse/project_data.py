import numpy as np
import os
import pandas as pd
from jinja2 import Template
from joblib import dump, Parallel, delayed
from os.path import join
from sklearn.utils import check_random_state

from cogspaces.datasets.utils import get_output_dir, get_data_dir
from cogspaces.plotting import plot_all
from cogspaces.utils import get_dictionary, get_masker
from exps.analyse.plot_maps import get_components
from exps.train import load_data


def compute_projections(layer='first', output_dir=None, n_jobs=3):
    data, target = load_data(join(get_data_dir(), 'reduced_512_gm'),
                             'all')

    first_proj = get_dictionary()
    if layer == 'first':
        full_proj = first_proj
        proj_dir = join(get_output_dir(), 'projected_512_gm')
        second_proj = None
    elif layer == 'second':
        second_proj = get_components(output_dir, return_type='arrays')
        full_proj = second_proj.dot(first_proj)
        proj_dir = join(output_dir, 'proj')
    else:
        raise ValueError()
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)
    gram = full_proj.dot(full_proj.T)
    inv_full_proj = np.linalg.inv(gram).dot(full_proj)

    Parallel(n_jobs=n_jobs)(delayed(compute_single)
                            (study, data[study], proj_dir, target[study],
                             inv_full_proj, second_proj)
                            for study in data)


def compute_single(study, this_data, proj_dir,
                   this_target, inv_full_proj, second_proj):
    filename = join(proj_dir, 'data_%s.pt' % study)

    if second_proj is None:
        y = this_data
    else:
        y = this_data.dot(second_proj.T)
    data_rec = y.dot(inv_full_proj)
    dump((data_rec, this_target), filename)


def pick_random_contrast():
    drop_dir = join(get_output_dir(), 'projections')
    if not os.path.exists(drop_dir):
        os.makedirs(drop_dir)

    masker = get_masker()

    data_dir = join(get_data_dir(), 'masked')
    data, _ = load_data(data_dir, 'all')

    first_proj_dir = join(get_output_dir(), 'projected_512_gm')
    data_proj, target = load_data(first_proj_dir, 'all')

    second_proj_dir = get_data_dir(
        join(get_output_dir(), 'full_model', 'proj'))
    data_proj_2, _ = load_data(second_proj_dir, 'all')

    data = {'raw': data, 'first': data_proj, 'second': data_proj_2}

    rng = check_random_state(37)
    selected_df = []
    selected = {'raw': [], 'first': [], 'second': []}
    for study, this_target in target.items():
        this_target['study'] = study
        this_target['index'] = np.arange(len(this_target), dtype='int')
        this_target = this_target.groupby(by=['study', 'contrast']).apply(
            lambda x:
            x.iloc[rng.randint(len(x))])
        selected_df.append(this_target)
        indices = this_target['index'].astype('int')
        for kind, this_selected in selected.items():
            this_selected.append(data[kind][study][indices])
    selected_df = pd.concat(selected_df, axis=0)
    selected_df.to_pickle(join(drop_dir, 'components_df.pkl'))

    dictionary = get_dictionary()
    dictionary_mask = np.any(dictionary, axis=0)

    for kind, this_selected in selected.items():
        components = np.concatenate(selected[kind], axis=0)
        if kind == 'raw':
            components *= dictionary_mask[None, :]
        print(components.shape)
        img = masker.inverse_transform(components)
        img.to_filename(join(drop_dir, 'components_%s.nii.gz' % kind))


def plot_contrast(n_jobs=3):
    drop_dir = join(get_output_dir(), 'projections')

    selected_df = pd.read_pickle(join(drop_dir, 'components_df.pkl'))
    full_names = ['%s::%s' % (study, contrast)
                  for study, contrast in selected_df[['study', 'contrast']].values]
    for kind in ['raw', 'first', 'second']:
        plot_all(join(drop_dir, 'components_%s.nii.gz' % kind),
                 output_dir=join(drop_dir, 'components_%s' % kind),
                 names=full_names,
                 threshold=False,
                 colors=None,
                 view_types=['stat_map'],
                 n_jobs=n_jobs)


def proj_html():
    drop_dir = join(get_output_dir(), 'projections')

    selected_df = pd.read_pickle(join(drop_dir, 'components_df.pkl'))

    metrics = pd.read_pickle(join(get_output_dir(), 'metrics', 'metrics.pkl'))
    baseline = pd.read_pickle(join(get_output_dir(), 'baseline', 'metrics.pkl'))
    metrics = metrics.loc[1e-4]
    selected_df['bacc'] = metrics['bacc'].groupby(['study', 'contrast']).mean()
    selected_df['gain'] = (metrics['bacc']
                           - baseline['bacc']).groupby(['study', 'contrast']).mean()
    selected_df = selected_df.sort_values(by='gain', ascending=False)
    full_names = ['%s::%s' % (study, contrast)
                  for study, contrast in selected_df[['study', 'contrast']].values]

    with open('plot_maps.html', 'r') as f:
        template = f.read()
    template = Template(template)
    imgs = []

    for name, (bacc, gain) in zip(full_names, selected_df[['bacc', 'gain']].values):
        view_types = ['stat_map']
        srcs = []
        for kind in ['raw', 'first', 'second']:
            components_dir = join(drop_dir, 'components_%s' % kind)
            for view_type in view_types:
                src = join(components_dir, '%s_%s.png' % (name, view_type))
                srcs.append(src)
        imgs.append((srcs, name + ' bacc=%.2f gain=%.2f' % (bacc, gain)))
    html = template.render(imgs=imgs)
    output_file = join(drop_dir, 'proj.html')
    with open(output_file, 'w+') as f:
        f.write(html)


if __name__ == '__main__':
    # compute_projections('first', n_jobs=20)
    # compute_projections('second', join(get_output_dir(), 'full_model'), n_jobs=20)
    # pick_random_contrast()
    plot_contrast(50)
    proj_html()