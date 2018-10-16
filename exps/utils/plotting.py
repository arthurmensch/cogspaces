"""
Plot functions for train.py
"""
import json
from os.path import join

import numpy as np
from joblib import load, dump
from seaborn import hls_palette

from cogspaces.plotting.volume import plot_4d_image
from cogspaces.report import compute_nifti, compute_grades, compute_names


# HTML report
def components_html(components_dir, output_dir, plot_wordclouds=True):
    from jinja2 import Template
    with open(join('assets', 'maps.html'), 'r') as f:
        template = f.read()
    template = Template(template)
    imgs = []
    for i in range(128):
        title = 'components_%i' % i
        view_types = ['stat_map', 'glass_brain', ]
        srcs = []
        for view_type in view_types:
            src = join(components_dir, '%s_%s.png' % (title, view_type))
            srcs.append(src)
        if plot_wordclouds:
            for grade_type in ['cosine_similarities']:
                wc_dir = 'wc_%s' % grade_type
                srcs.append(join(wc_dir, 'wc_single_%i.png' % i))
                srcs.append(join(wc_dir, 'wc_cat_%i.png' % i))
        imgs.append((srcs, title))
    html = template.render(imgs=imgs)
    output_file = join(output_dir, 'components.html')
    with open(output_file, 'w+') as f:
        f.write(html)


def classifs_html(full_names, classifs_dir, output_dir):
    from jinja2 import Template
    with open(join('assets', 'maps.html'), 'r') as f:
        template = f.read()
    template = Template(template)
    imgs = []
    for name in full_names:
        view_types = ['stat_map', 'glass_brain',
                      ]
        srcs = []
        for view_type in view_types:
            src = join(classifs_dir, '%s_%s.png' % (name, view_type))
            srcs.append(src)
        imgs.append((srcs, name))
    html = template.render(imgs=imgs)
    output_file = join(output_dir, 'classifs.html')
    with open(output_file, 'w+') as f:
        f.write(html)


def make_plots(output_dir, plot_components=True, plot_classifs=True,
         plot_surface=True, plot_wordclouds=True, n_jobs=1):
    """Plot all qualitative figures from model record.
    """
    names = load(join(output_dir, 'names.pkl'))
    full_names = load(join(output_dir, 'full_names.pkl'))

    # Colors
    colors = np.arange(128)
    colors_2d = np.array(hls_palette(128, s=1, l=.4))
    colors_3d = np.array(hls_palette(128, s=1, l=.5))
    np.save(join(output_dir, 'colors_2d.npy'), colors_2d)
    np.save(join(output_dir, 'colors_3d.npy'), colors_3d)

    # 2D plots
    if plot_classifs:
        classifs_imgs = join(output_dir, 'classifs.nii.gz')
        view_types = ['stat_map', 'glass_brain', ]
        plot_4d_image(classifs_imgs,
                      output_dir=join(output_dir, 'classifs'),
                      names=full_names,
                      view_types=view_types, threshold=0,
                      n_jobs=n_jobs)
        classifs_html(full_names, 'classifs', output_dir)
    if plot_components:
        components_imgs = join(output_dir, 'components.nii.gz')
        plot_4d_image(components_imgs,
                      output_dir=join(output_dir, 'components'),
                      names='components',
                      colors=colors_2d,
                      view_types=view_types,
                      n_jobs=n_jobs)

    if plot_surface:
        # 3D plots
        from cogspaces.plotting.surface import plot_4d_image_surface
        plot_4d_image_surface(components_imgs, colors_3d, output_dir)

    # Wordclouds
    if plot_wordclouds:
        from cogspaces.plotting.wordclouds import plot_word_clouds
        grades = load(join(output_dir, 'grades.pkl'))
        plot_word_clouds(join(output_dir, 'wc'), grades, n_jobs=n_jobs,
                         colors=colors)

    components_html('components', output_dir, plot_wordclouds=plot_wordclouds)


def prepare_plots(output_dir, make_grades=True):
    target_encoder = load(join(output_dir, 'target_encoder.pkl'))
    estimator = load(join(output_dir, 'estimator.pkl'))
    with open(join(output_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    if config['model']['normalize']:
        standard_scaler = load(join(output_dir, 'standard_scaler.pkl'))
    else:
        standard_scaler = None

    niftis = compute_nifti(estimator, standard_scaler, config)

    if config['model']['estimator'] in ['multi_study', 'ensemble']:
        classifs_img, components_imgs = niftis
        classifs_img.to_filename(join(output_dir, 'classifs.nii.gz'))
        components_imgs.to_filename(join(output_dir, 'components.nii.gz'))
        if make_grades:
            grades = compute_grades(estimator, standard_scaler, target_encoder,
                                    config, grade_type='cosine_similarities', )
            dump(grades, join(output_dir, 'grades.pkl'))
    else:
        classifs_img = niftis
        classifs_img.to_filename(join(output_dir, 'classifs.nii.gz'))

    names, full_names = compute_names(target_encoder)
    dump(names, join(output_dir, 'names.pkl'))
    dump(full_names, join(output_dir, 'full_names.pkl'))