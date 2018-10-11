"""
Utility function for train.py
"""
import json
from os.path import join

import numpy as np
from cogspaces.plotting.volume import plot_4d_image
from cogspaces.report import compute_names, compute_nifti, compute_grades
from joblib import load, dump
from seaborn import hls_palette


# HTML report
def components_html(output_dir, components_dir, plot_wordclouds=True):
    from jinja2 import Template
    with open('plot_maps.html', 'r') as f:
        template = f.read()
    template = Template(template)
    imgs = []
    for i in range(128):
        title = 'components_%i' % i
        view_types = ['stat_map', 'glass_brain',]
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


def classifs_html(output_dir, classifs_dir):
    from jinja2 import Template
    with open('plot_maps.html', 'r') as f:
        template = f.read()
    names, full_names = compute_names(output_dir)
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


def plot(output_dir, plot_components=True,
         plot_classifs=True,
         plot_surface=True, plot_wordclouds=True,
         n_jobs=1):

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
        classifs_html(output_dir, 'classifs')
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

    components_html(output_dir, 'components', plot_wordclouds=plot_wordclouds)


def save(estimator, standard_scaler, target_encoder, metrics, info, config,
         output_dir, save_grades=True):
    dump(target_encoder, join(output_dir, 'target_encoder.pkl'))
    dump(standard_scaler, join(output_dir, 'standard_scaler.pkl'))
    dump(estimator, join(output_dir, 'estimator.pkl'))
    dump(metrics, join(output_dir, 'metrics.pkl'))
    with open(join(output_dir, 'info.json'), 'w+') as f:
        json.dump(info, f)
    with open(join(output_dir, 'config.json'), 'w+') as f:
        json.dump(config, f)
    niftis = compute_nifti(estimator, standard_scaler, config)

    if config['model']['estimator'] in ['factored', 'ensemble']:
        classifs_img, components_imgs = niftis
        classifs_img.to_filename(output_dir, 'classifs.nii.gz')
        components_imgs.to_filename(output_dir, 'components_imgs.nii.gz')
        if save_grades:
            grades = compute_grades(output_dir,
                                    grade_type='cosine_similarities')
            dump(grades, join(output_dir, 'grades.pkl'))
    else:
        classifs_img = niftis
        classifs_img.to_filename(output_dir, 'classifs.nii.gz')

    names, full_names = compute_names(target_encoder)
    dump(names, join(output_dir, 'names.pkl'))
    dump(full_names, join(output_dir, 'full_names.pkl'))
