#! /usr/bin/env python3
from collections import defaultdict
from itertools import repeat

import numpy as np
import os
import re
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
from nilearn import surface
from nilearn._utils import check_niimg
from nilearn.datasets import fetch_surf_fsaverage5
from nilearn.image import iter_img
from os.path import join
from wordcloud import WordCloud


def make_cmap(color, rotation=.5, white=False):
    h, s, v = rgb_to_hsv(color)
    h = h + rotation
    if h > 1:
        h -= 1
    r, g, b = color
    ri, gi, bi = hsv_to_rgb((h, s, v))
    colors = {'direct': (ri, gi, bi), 'inverted': (r, g, b)}
    cdict = {}
    for direction, (r, g, b) in colors.items():
        if white:
            cdict[direction] = {color: [(0.0, 0.0416, 0.0416),
                  (0.18, c, c),
                  (0.5, 1, 1),
                  (0.62, 0.0, 0.0),
                  (1.0, 0.0416, 0.0416)] for color, c in [('blue', b), ('red', r), ('green', g)]}
            cdict[direction]['alpha']: [(0, 1, 1), (0.5, 0, 0), (1, 1, 1)]
        else:
            cdict[direction] = {color: [(0.0, 1, 1),
                  (0.32, c, c),
                  (0.5, 0.0416, 0.0416),
                  (0.5, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 1, 1)]  for color, c in [('blue', b), ('red', r), ('green', g)]}
    cmap = LinearSegmentedColormap('cmap', cdict['direct'])
    cmapi = LinearSegmentedColormap('cmap', cdict['inverted'])
    cmap._init()
    cmapi._init()
    cmap._lut = np.maximum(cmap._lut, cmapi._lut[::-1])
    cmap._lut[-1, -1] = 0
    return cmap


def plot_single(img, name, output_dir, view_types=['stat_map'], color=None):
    import matplotlib
    matplotlib.use('agg')
    from nilearn.plotting import plot_stat_map, find_xyz_cut_coords, \
        plot_glass_brain, plot_surf_stat_map

    if color is not None:
        cmap = make_cmap(color, rotation=.5)
        cmap_white = make_cmap(color, rotation=.5, white=True)
    else:
        cmap = 'cold_hot'
        cmap_white = 'cold_white_hot'

    srcs = []
    vmax = np.abs(img.get_data()).max()

    for view_type in view_types:
        src = join(output_dir, '%s_%s.png' % (name, view_type))
        if view_type in ['surf_stat_map_lateral_right',
                         'surf_stat_map_lateral_left',
                         'surf_stat_map_medial_right',
                         'surf_stat_map_medial_left']:
            fsaverage = fetch_surf_fsaverage5()
            view = 'lateral' if 'lateral' in view_type else 'medial'

            if 'right' in view_type:
                surf_mesh = fsaverage.infl_right
                bg_map = fsaverage.sulc_right
                texture_surf_mesh = fsaverage.pial_right
                texture = surface.vol_to_surf(img, texture_surf_mesh)
                hemi = 'right'
            else:
                surf_mesh = fsaverage.infl_left
                bg_map = fsaverage.sulc_left
                texture_surf_mesh = fsaverage.pial_left
                texture = surface.vol_to_surf(img, texture_surf_mesh)
                hemi = 'left'
            plot_surf_stat_map(surf_mesh, texture, hemi=hemi,
                               bg_map=bg_map,
                               vmax=vmax,
                               threshold=vmax / 12,
                               view=view,
                               output_file=src,
                               cmap=cmap)

        elif view_type in ['stat_map', 'glass_brain']:
            cut_coords = find_xyz_cut_coords(img,
                                             activation_threshold=vmax / 3)
            if view_type == 'stat_map':
                plot_stat_map(img, threshold=vmax / 6, cut_coords=cut_coords,
                              vmax=vmax,
                              colorbar=True, output_file=src, cmap=cmap)
            else:
                plot_glass_brain(img, threshold=vmax / 6,
                                 cut_coords=cut_coords,
                                 vmax=vmax,
                                 plot_abs=False, output_file=src,
                                 colorbar=True,
                                 cmap='cold_white_hot')
        else:
            raise ValueError('Wrong view type in `view_types`: got %s' %
                             view_type)
        srcs.append(src)
    return srcs, name


def numbered_names(name):
    i = 0
    while True:
        yield '%s_%i' % (name, i)
        i += 1


def plot_all(img, names=None, output_dir=None,
             colors=None,
             view_types=['stat_map'],
             n_jobs=1, verbose=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if colors is None:
        colors = repeat(None)

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
    imgs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(plot_single)(img, name, output_dir, view_types, color)
        for name, img, color in zip(names, iter_img(img), colors))
    return imgs


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


def filter_contrast(contrast):
    contrast = contrast.lower()
    contrast = contrast.replace('lf', 'left foot')
    contrast = contrast.replace('rh', 'right foot')
    contrast = contrast.replace('lh', 'left hand')
    contrast = contrast.replace('rh', 'right hand')
    contrast = contrast.replace('clicgaudio', 'left audio click')
    contrast = contrast.replace('clicgvideo', 'left video click')
    contrast = contrast.replace('clicdvideo', 'left video click')
    contrast = contrast.replace('clicdaudio', 'right audio click')
    contrast = contrast.replace('calculvideo', 'video calculation')
    contrast = contrast.replace('calculaudio', 'audio calculation')

    contrast = contrast.replace('audvid600', 'audio video 600ms')
    contrast = contrast.replace('audvid1200', 'audio video 1200ms')
    contrast = contrast.replace('audvid300', 'audio video 300ms')
    contrast = contrast.replace('bk', 'back')
    contrast = contrast.replace('realrt', 'real risk-taking')
    contrast = contrast.replace('rt', 'risk-taking')
    contrast = contrast.replace('reapp', 'reappraise')
    contrast = re.sub(r'\b(neu)\b', 'neutral', contrast)
    contrast = re.sub(r'\b(neg)\b', 'negative', contrast)
    contrast = re.sub(r'\b(ant)\b', 'anticipated', contrast)
    return contrast


def plot_word_clouds_all(output_dir, grades):
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
            contrast = filter_contrast()
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
                frequencies.append((contrast, grade))
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


def plot_word_clouds(output_dir, grades, f1s=None, n_jobs=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Parallel(n_jobs=n_jobs, verbose=10)(delayed(plot_word_cloud_single)
                                        (output_dir, grades, i, f1s)
                                        for i, grades in
                                        enumerate(grades['full']))


def plot_word_cloud_single(output_dir, grades, index, f1s=None):
    import matplotlib.pyplot as plt

    contrasts = list(filter(
        lambda x: 'effects_of_interest' not in x and 'gauthier' not in x,
        grades))[:15]
    frequencies_cat = defaultdict(lambda: 0.)
    frequencies_single = defaultdict(lambda: 0.)
    for contrast in contrasts:
        grade = grades[contrast]
        study, contrast = contrast.split('::')
        f1 = 1 if f1s is None else f1s[study][contrast]
        contrast = filter_contrast(contrast)
        terms = contrast.replace(' ', '_').replace('&', '_'). \
            replace('-', '_').split('_')
        cat_terms = []
        for term in terms:
            if term == 'baseline':
                break
            if term == 'vs':
                break
            cat_terms.append(term)
        for term in cat_terms:
            frequencies_single[term] += grade * f1 / len(cat_terms)
        cat_terms = ' '.join(cat_terms)
        frequencies_cat[cat_terms] += grade * f1

    dpi = 40
    width, height = (400, 200)
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi))
    fig, ax = plt.subplots(1, 1)
    wc = WordCloud(prefer_horizontal=1,
                   background_color='white',
                   relative_scaling=0.7)
    wc.generate_from_frequencies(frequencies=frequencies_single, )
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.savefig(join(output_dir, 'wc_single_%i.png' % index))
    plt.close(fig)
    width, height = (600, 200)
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi))
    wc = WordCloud(prefer_horizontal=1,
                   background_color='white', width=800, height=200,
                   relative_scaling=0.7)
    wc.generate_from_frequencies(frequencies=frequencies_cat, )
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.savefig(join(output_dir, 'wc_cat_%i.png' % index))
    plt.close(fig)
