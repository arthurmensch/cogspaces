import os

import numpy as np
from jinja2 import Template
from joblib import delayed, Parallel, load
from nilearn._utils import check_niimg
from nilearn.image import iter_img
from nilearn.plotting import find_xyz_cut_coords, plot_stat_map, \
    plot_glass_brain
from os.path import join

from cogspaces.datasets.utils import get_output_dir

import pandas as pd


def plot_single_tuple(imgs, view_type, plot_dir, name):
    n_imgs = len(imgs)
    if view_type == 'stat_map':
        vmax = np.abs(imgs[0].get_data()).max()
        cut_coords = find_xyz_cut_coords(imgs[0],
                                         activation_threshold=vmax / 3)
        for i, img in enumerate(imgs):
            src = join(plot_dir, '%s_%s_%i.png' % (name, view_type, i))
            this_vmax = np.abs(img.get_data()).max()
            plot_stat_map(img, threshold=0,
                          cut_coords=cut_coords,
                          vmax=this_vmax,
                          colorbar=True,
                          output_file=src)
            plot_stat_map(img, threshold=0,
                          cut_coords=cut_coords,
                          vmax=this_vmax,
                          colorbar=True,
                          output_file=src.replace('.png', '.svg'), )
    else:
        for i, img in enumerate(imgs):
            src = join(plot_dir, '%s_%s_%i.png' % (name, view_type, i))
            this_vmax = np.abs(img.get_data()).max()
            plot_glass_brain(img, threshold=this_vmax / 3,
                             vmax=this_vmax,
                             plot_abs=False,
                             output_file=src,
                             colorbar=True,
                             )
            plot_glass_brain(img, threshold=this_vmax / 3,
                             vmax=this_vmax,
                             plot_abs=False,
                             output_file=src.replace('.png', '.svg'),
                             colorbar=True,
                             )


def multi_iter_imgs(imgs_list):
    n = imgs_list[0].shape[3]
    imgs_iter_list = [iter_img(imgs) for imgs in imgs_list]
    for i in range(n):
        yield [next(imgs_iter) for imgs_iter in imgs_iter_list]


def plot_all(components_list, names, plot_dir, n_jobs=4):
    view_types = ['stat_map', 'glass_brain']

    components_list = [check_niimg(components)
                       for components in components_list]
    imgs_list = multi_iter_imgs(components_list)

    Parallel(n_jobs=n_jobs, verbose=10)(delayed(plot_single_tuple)
                            (imgs, view_type, plot_dir, name)
                            for imgs, name in zip(imgs_list, names)
                            for view_type in view_types)


def classifs_html(output_dir):
    with open('plot_maps.html', 'r') as f:
        template = f.read()
    template = Template(template)
    imgs = []
    plot_dir = 'imgs'
    view_types = ['stat_map']

    # Sort by performance gain
    metrics = pd.read_pickle(join(get_output_dir(), 'figure_4',
                                  'metrics', 'metrics.pkl'))
    baseline = pd.read_pickle(
        join(get_output_dir(), 'figure_4',
             'metrics', 'baseline', 'metrics.pkl'))
    metrics = metrics.loc[1e-4]
    bacc = metrics['bacc'].groupby(['study', 'contrast']).mean()
    gain = (metrics['bacc'] - baseline['bacc']).groupby(
        ['study', 'contrast']).mean()
    df = pd.DataFrame(dict(gain=gain, bacc=bacc))
    df = df.sort_values(by='gain', ascending=False)
    df = df.reset_index()

    for row in df.itertuples():
        name = '%s::%s' % (row.study, row.contrast)
        srcs = []
        for i in [2, 1, 0]:
            for view_type in view_types:
                src = join(plot_dir, '%s_%s_%i.png' % (name, view_type, i))
                srcs.append(src)
        imgs.append((srcs, name + '\tbacc=%.3f\tgain=%.3f' % (row.bacc, row.gain)))
    html = template.render(imgs=imgs)
    output_file = join(output_dir, 'imgs.html')
    with open(output_file, 'w+') as f:
        f.write(html)


def classifs_maps():
    output_dir = join(get_output_dir(), 'figure_4', 'classifs')
    # components_list = ['classifs_factored.nii.gz',
    #                    'classifs_logistic.nii.gz',
    #                    'classifs_full.nii.gz']
    # components_list = [join(output_dir, components)
    #                    for components in components_list]
    # names = load(join(output_dir, 'names.pkl'))
    # plot_dir = join(output_dir, 'imgs')
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plot_all(components_list, names, plot_dir, n_jobs=45)
    classifs_html(output_dir)


def projections():
    output_dir = join(get_output_dir(), 'figure_4', 'projections')
    # components_list = ['components_second.nii.gz',
    #                    'components_first.nii.gz',
    #                    'components_raw.nii.gz'
    #                    ]
    # components_list = [join(output_dir, components)
    #                    for components in components_list]
    # names = load(join(output_dir, 'names.pkl'))
    # plot_dir = join(output_dir, 'imgs')
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plot_all(components_list, names, plot_dir, n_jobs=45)
    classifs_html(output_dir)

projections()
classifs_maps()
