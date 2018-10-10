import os
from itertools import repeat
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
from nilearn import surface
from nilearn._utils import check_niimg
from nilearn.datasets import fetch_surf_fsaverage5
from nilearn.image import iter_img
from nilearn.input_data import NiftiMasker

from cogspaces.datasets import fetch_mask


def make_cmap(color, rotation=.5, white=False, transparent_zero=False):
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
                                        (1.0, 0.0416, 0.0416)] for color, c in
                                [('blue', b), ('red', r), ('green', g)]}
        else:
            cdict[direction] = {color: [(0.0, 1, 1),
                                        (0.32, c, c),
                                        (0.5, 0.0416, 0.0416),
                                        (0.5, 0.0, 0.0),
                                        (0.87, 0.0, 0.0),
                                        (1.0, 1, 1)] for color, c in
                                [('blue', b), ('red', r), ('green', g)]}
        if transparent_zero:
            cdict[direction]['alpha']: [(0, 1, 1), (0.5, 0, 0), (1, 1, 1)]
    cmap = LinearSegmentedColormap('cmap', cdict['direct'])
    cmapi = LinearSegmentedColormap('cmap', cdict['inverted'])
    cmap._init()
    cmapi._init()
    cmap._lut = np.maximum(cmap._lut, cmapi._lut[::-1])
    # Big hack from nilearn (WTF !?)
    cmap._lut[-1, -1] = 0
    return cmap


def plot_single(img, name, output_dir, view_types=['stat_map'], color=None,
                threshold=0):
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
    threshold = vmax / 8

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
                               threshold=threshold,
                               view=view,
                               output_file=src,
                               cmap=cmap)

        elif view_type in ['stat_map', 'glass_brain']:
            cut_coords = find_xyz_cut_coords(img,
                                             activation_threshold=vmax / 3)
            if view_type == 'stat_map':
                plot_stat_map(img, threshold=threshold,
                              cut_coords=cut_coords,
                              vmax=vmax,
                              colorbar=False,
                              output_file=src,
                              # cmap=cmap
                              )
                plot_stat_map(img, threshold=threshold,
                              cut_coords=cut_coords,
                              vmax=vmax,
                              display_mode='ortho',
                              colorbar=True,
                              output_file=src.replace('.png', '_z.svg'),
                              # cmap=cmap
                              )
            else:
                plot_glass_brain(img, threshold=threshold,
                                 vmax=vmax,
                                 plot_abs=False,
                                 output_file=src,
                                 colorbar=False,
                                 # cmap=cmap_white
                                 )
                plot_glass_brain(img, threshold=threshold,
                                 vmax=vmax,
                                 display_mode='ortho',
                                 plot_abs=False,
                                 output_file=src.replace('.png', '_xz.svg'),
                                 colorbar=True,
                                 # cmap=cmap_white
                                 )
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


def plot_4d_image(img, names=None, output_dir=None,
                  colors=None,
                  view_types=['stat_map'],
                  threshold=True,
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

    mask = fetch_mask()
    masker = NiftiMasker(mask_img=mask).fit()
    components = masker.transform(img)
    n_components = len(components)
    threshold = np.percentile(np.abs(components),
                              100. * (1 - 1. / n_components)) if threshold else 0

    imgs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(plot_single)(img, name, output_dir, view_types, color,
                             threshold=threshold)
        for name, img, color in zip(names, iter_img(img), colors))
    return imgs