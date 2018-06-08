#! /usr/bin/env python3

import numpy as np
import os
from joblib import Parallel, delayed
from nilearn._utils import check_niimg
from nilearn.image import iter_img
from os.path import join


def plot_single(img, name, output_dir):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_stat_map, find_xyz_cut_coords, plot_glass_brain

    src, glass_src = '%s.png' % name, '%s_glass.png' % name
    vmax = np.abs(img.get_data()).max()
    cut_coords = find_xyz_cut_coords(img, activation_threshold=vmax / 3)
    fig = plt.figure(figsize=(8, 8))
    plot_stat_map(img, figure=fig, threshold=0, cut_coords=cut_coords,
                  colorbar=False)
    plt.savefig(join(output_dir, name))
    plt.close(fig)
    fig = plt.figure(figsize=(8, 8))
    plot_glass_brain(img, figure=fig, threshold=0, cut_coords=cut_coords,
                     plot_abs=False)
    plt.savefig(join(output_dir, glass_src))
    plt.close(fig)
    return src, glass_src


def plot_all(imgs, output_dir, name, n_jobs=1):
    img_dir = join(output_dir, 'imgs')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    imgs = check_niimg(imgs)
    srcs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(plot_single)(img,
                             ('%s_%i' % (name, i)), img_dir)
        for i, img in
        enumerate(iter_img(imgs)))
    with open(join(output_dir, '%s.html' % name), 'w+') as f:
        f.write("""<html><head><title>%s</title></head>\n<body>\n
                <h1>%s</h1>""" % (name, name))
        for src, glass_src in srcs:
            f.write("""<p><img src='imgs/%s'>\n<img src='imgs/%s'></p>\n"""
                    % (src, glass_src))
        f.write("""</body>""")
