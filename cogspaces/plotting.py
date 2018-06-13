#! /usr/bin/env python3
from itertools import repeat

import numpy as np
import os
from joblib import Parallel, delayed
from nilearn import surface
from nilearn._utils import check_niimg
from nilearn.datasets import fetch_surf_fsaverage5
from nilearn.image import iter_img
from os.path import join
from wordcloud import WordCloud


def plot_single(img, name, output_dir, view_types=['stat_map']):
    import matplotlib
    matplotlib.use('agg')
    from nilearn.plotting import plot_stat_map, find_xyz_cut_coords,\
        plot_glass_brain, plot_surf_stat_map

    srcs = []
    for view_type in view_types:
        src = join(output_dir, '%s_%s.png' % (name, view_type))
        if view_type in ['surf_stat_map_right', 'surf_stat_map_left']:
            fsaverage = fetch_surf_fsaverage5()
            vmax = np.abs(img.get_data()).max()
            if view_type == 'surf_stat_map_right':
                texture = surface.vol_to_surf(img, fsaverage.pial_right)
                plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                                   bg_map=fsaverage.sulc_right, threshold=0,
                                   vmax=vmax,
                                   output_file=join(output_dir, src),
                                   cmap='cold_hot')
            else:
                texture = surface.vol_to_surf(img, fsaverage.pial_left)
                plot_surf_stat_map(fsaverage.infl_left, texture, hemi='left',
                                   bg_map=fsaverage.sulc_right, threshold=0,
                                   vmax=vmax,
                                   output_file=join(output_dir, src),
                                   cmap='cold_hot')

        elif view_type in ['stat_map', 'glass_brain']:
            vmax = np.abs(img.get_data()).max()
            cut_coords = find_xyz_cut_coords(img, activation_threshold=vmax / 3)
            if view_type == 'stat_map':
                plot_stat_map(img, threshold=0, cut_coords=cut_coords,
                              colorbar=False, output_file=join(output_dir, src))
            else:
                plot_glass_brain(img, threshold=0, cut_coords=cut_coords,
                                 plot_abs=False, output_file=join(output_dir, src))
        else:
            raise ValueError('Wrong view type in `view_types`: got %s' % view_type)
        srcs.append(src)
    return srcs


def numbered_names(name):
    i = 0
    while True:
        yield '%s_%i' % (name, i)
        i += 1


def make_html(imgs, output_dir, name, filename=None,
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


def plot_all(img, names=None, output_dir=None,
             view_types=['stat_map'],
             n_jobs=1, verbose=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if 'surf_stat_map_right' in view_types or 'surf_stat_map_left' in view_types:
        fetch_surf_fsaverage5()
    filename = img
    img = check_niimg(img, ensure_ndim=4)
    img.get_data()
    if names is None or isinstance(names, str):
        if names is None:
            dirname, filename = os.path.split(filename)
            names = filename.replace('.nii.gz', '')
        names = numbered_names(names)
    else:
        assert len(names) == img.get_shape()[3]
    Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(plot_single)(img, name, output_dir, view_types)
        for name, img in zip(names, iter_img(img)))


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


def rgb2hex(r, g, b):
    return f'#{int(round(r * 255)):02x}' \
           f'{int(round(g * 255)):02x}' \
           f'{int(round(b * 255)):02x}'


def plot_word_clouds(output_dir, grades):
    import seaborn as sns
    import matplotlib.pyplot as plt

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    studies = list(grades['study'][0].keys())
    colors = sns.color_palette('husl', len(studies))

    for i, these_grades in enumerate(grades['full']):
        contrasts = list(filter(
            lambda x: 'effects_of_interest' not in x and 'gauthier' not in x,
            these_grades))[:15]
        frequencies = []
        studies = []
        for contrast in contrasts:
            grade = these_grades[contrast]
            study, contrast = contrast.split('::')
            if study == 'hcp':
                contrast = contrast.replace('LF', 'left foot')
                contrast = contrast.replace('RF', 'right foot')
                contrast = contrast.replace('LH', 'left hand')
                contrast = contrast.replace('RH', 'right hand')
            contrast = contrast.replace('clicGaudio', 'left audio click')
            contrast = contrast.replace('clicDaudio', 'right audio click')
            contrast = contrast.replace('calculvideo', 'video calculation')
            contrast = contrast.replace('calculaudio', 'audio calculation')

            terms = contrast.split('_')
            contrast = []
            for term in terms:
                if term == 'baseline':
                    break
                if term == 'vs':
                    break
                else:
                    contrast.append(term)
            if contrast:
                contrast = ' '.join(contrast)
                curated = contrast.lower()
                frequencies.append((curated, grade))
                studies.append(study)
        color_to_words = {rgb2hex(*color): [study]
                          for color, study in zip(colors, studies)}
        color_func = SimpleGroupedColorFunc(color_to_words,
                                            default_color='#ffffff')
        wc = WordCloud(color_func=color_func)
        wc.generate_from_frequencies(frequencies=frequencies,
                                     as_tuples=True,
                                     group_colors=studies)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.savefig(join(output_dir, 'wc_%i.png' % i))
        plt.close(fig)