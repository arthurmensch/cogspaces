#! /usr/bin/env python3
from itertools import repeat

import numpy as np
import os
from joblib import Parallel, delayed
from nilearn._utils import check_niimg
from nilearn.image import iter_img
from os.path import join


def plot_single(img, name, output_dir, draw=True):
    src, glass_src = '%s.png' % name, '%s_glass.png' % name

    if draw:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        from nilearn.plotting import plot_stat_map, find_xyz_cut_coords, plot_glass_brain



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


def numbered_names(name):
    i = 0
    while True:
        yield '%s_%i' % (name, i)
        i += 1


def plot_all(imgs, output_dir, name, filename=None,
             texts=None,
             word_clouds=False,
             names=None, n_jobs=1, verbose=10,
             draw=True):
    img_dir = join(output_dir, 'imgs')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if names is None:
        names = numbered_names(name)
    if texts is None:
        texts = repeat('')
    if filename is None:
        filename = name

    imgs = check_niimg(imgs)
    srcs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(plot_single)(img, name, img_dir, draw)
        for name, img in zip(names, iter_img(imgs)))
    with open(join(output_dir, '%s.html' % filename), 'w+') as f:
        f.write("""<html><head><title>%s</title></head>\n<body>\n
                <h1>%s</h1>""" % (name, name))
        for i, ((src, glass_src), text) in enumerate(zip(srcs, texts)):
            f.write(text)
            f.write("""<p>""")
            f.write("""<img src='imgs/%s'>\n<img src='imgs/%s'>\n"""
                    % (src, glass_src))
            f.write("""<img src='wc/wc_%i.png'>\n""" % i)
            f.write("""</p>""")

        f.write("""</body>""")
