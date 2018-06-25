import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from jinja2 import Template
from joblib import delayed, Parallel, load
from matplotlib import transforms
from nibabel.tests.test_viewers import matplotlib
from nilearn._utils import check_niimg
from nilearn.image import iter_img, index_img
from nilearn.plotting import find_xyz_cut_coords, plot_stat_map, \
    plot_glass_brain, find_cut_slices
from os.path import join

from cogspaces.datasets.utils import get_output_dir


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
        imgs.append(
            (srcs, name + '\tbacc=%.3f\tgain=%.3f' % (row.bacc, row.gain)))
    html = template.render(imgs=imgs)
    output_file = join(output_dir, 'imgs.html')
    with open(output_file, 'w+') as f:
        f.write(html)


def classifs_maps():
    output_dir = join(get_output_dir(), 'figure_4', 'classifs')
    components_list = ['classifs_factored.nii.gz',
                       'classifs_logistic.nii.gz',
                       'classifs_full.nii.gz']
    components_list = [join(output_dir, components)
                       for components in components_list]
    names = load(join(output_dir, 'names.pkl'))
    plot_dir = join(output_dir, 'imgs')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_all(components_list, names, plot_dir, n_jobs=45)
    classifs_html(output_dir)


def projections():
    output_dir = join(get_output_dir(), 'figure_4', 'projections')
    components_list = ['components_second.nii.gz',
                       'components_first.nii.gz',
                       'components_raw.nii.gz'
                       ]
    components_list = [join(output_dir, components)
                       for components in components_list]
    names = load(join(output_dir, 'names.pkl'))
    plot_dir = join(output_dir, 'imgs')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_all(components_list, names, plot_dir, n_jobs=45)
    classifs_html(output_dir)


def plot_classifs_selection():
    gs = gridspec.GridSpec(5, 3, width_ratios=[2, 2, 1.25],
                           hspace=0., wspace=0.04)
    fig = plt.figure(figsize=(9, 6))

    fig.subplots_adjust(left=0.0, right=1., top=.905, bottom=0.01)

    classifs = [
                ('ds105', 'face_vs_house'),
                ('ds001', 'pumps_demean_vs_ctrl_demean'),
                ('pinel2009twins', 'language_vs_sound'),
                ('cauvet2009muslang', 'c16_c08_vs_c01_C02_music'),
                ('brainomics', 'vertical_checkerboard')]

    fig_names = ["Face vs house\nHaxby et al., '01",
                 "Pumps vs control\nSchonberg et al., '12",
                 "Language vs sound\nPinel et al., '09",
                 "Complex vs simple music\nCauvet et al., '09",
                 "Vertical checkerboard\nPapadopoulos et al., '15"]

    output_dir = join(get_output_dir(), 'figure_4')
    names = load(join(output_dir, 'classifs', 'names.pkl'))
    index = []
    for name in names:
        study, contrast = name.split('::')
        index.append((study, contrast))
    index = pd.MultiIndex.from_tuples(index)
    indices = pd.Series(data=np.arange(len(names), dtype='int'),
                        index=index, name='index')
    selected = indices.loc[classifs]

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

    classifs_factored = check_niimg(join(output_dir, 'classifs',
                                         'classifs_factored.nii.gz'))
    classifs_factored = index_img(classifs_factored, selected.values.tolist())
    classifs_full = check_niimg(join(output_dir, 'classifs',
                                     'classifs_full.nii.gz'))
    classifs_full = index_img(classifs_full, selected.values.tolist())
    fig.savefig(join(output_dir, 'figure_4.svg'))
    offset_ann = [6, 0, 0, 4, 0]
    for num, ((study, contrast), index) in enumerate(selected.iteritems()):
        sub_gs_full = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                       subplot_spec=gs[num, 0],
                                                       wspace=0.02,
                                                       width_ratios=[2, 1],
                                                       hspace=0.0)
        sub_gs_factored = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                           subplot_spec=
                                                           gs[num, 1],
                                                           wspace=0.02,
                                                           width_ratios=[2, 1],
                                                           hspace=0.0)
        ax_full = fig.add_subplot(sub_gs_full[0])
        ax_full_glass = fig.add_subplot(sub_gs_full[1])
        ax_factored = fig.add_subplot(sub_gs_factored[0])
        ax_factored_glass = fig.add_subplot(sub_gs_factored[1])
        ax_ann = fig.add_subplot(gs[num, 2])

        full_img = index_img(classifs_full, num)
        factored_img = index_img(classifs_factored, num)
        vmax = np.abs(factored_img.get_data()).max()
        vmax_full = np.abs(full_img.get_data()).max()
        if contrast == 'c16_c08_vs_c01_C02_music':
            cut_coords = (-52, 0, 42)
        elif contrast == 'language_vs_sound':
            cut_coords = (-54, 0, -8)
        else:
            cut_coords = find_xyz_cut_coords(factored_img,
                                             activation_threshold=vmax / 3)
        display_mode = 'xz' if num != 4 else 'yz'
        if num != 4:
            s_cut_coords = (cut_coords[0], cut_coords[2])
            ann = 'x = %i, z = %i' % s_cut_coords
        else:
            s_cut_coords = (cut_coords[1], cut_coords[2])
            ann = 'y = %i, z = %i' % s_cut_coords

        plot_stat_map(full_img, figure=fig,
                      threshold=0,
                      colorbar=False,
                      annotate=False,
                      cut_coords=s_cut_coords,
                      display_mode=display_mode,
                      vmax=vmax_full,
                      axes=ax_full)
        plot_glass_brain(full_img, figure=fig,
                         threshold=vmax_full / 3,
                         plot_abs=False,
                         vmax=vmax_full,
                         cut_coords=cut_coords,
                         colorbar=False,
                         annotate=False,
                         display_mode='z',
                         axes=ax_full_glass)
        plot_stat_map(factored_img, figure=fig,
                      threshold=0,
                      colorbar=False,
                      annotate=False,
                      display_mode=display_mode,
                      vmax=vmax,
                      cut_coords=s_cut_coords,
                      axes=ax_factored)
        plot_glass_brain(factored_img, figure=fig,
                         threshold=vmax / 3,
                         vmax=vmax,
                         colorbar=False,
                         cut_coords=cut_coords,
                         plot_abs=False,
                         display_mode='z',
                         axes=ax_factored_glass)
        ax_full.annotate(ann, xycoords='axes fraction',
                         va='bottom',
                         xytext=(0., offset_ann[num]),
                         textcoords='offset points',
                         bbox=dict(facecolor='white', edgecolor=None,
                                   linewidth=0, pad=0),
                         ha='left', xy=(.03, -0.04))

        if num == 3:
            cut_slices = find_cut_slices(full_img, n_cuts=7)
            plot_stat_map(full_img, display_mode='z', cut_coords=cut_slices,
                          output_file=join(output_dir,
                                           'axial_classification_vs_baseline_full.png'))
            plot_stat_map(factored_img, display_mode='z',
                          cut_coords=cut_slices,
                          output_file=join(output_dir,
                                           'axial_classification_vs_baseline_factored.png'))

        ax_ann.axis('off')
        this_gain = gain.loc[(study, contrast)]
        this_bacc = bacc.loc[(study, contrast)]
        ax_ann.annotate('%s\n'
                        'B-accuracy: %.1f%%\nB-acc. gain: %+.1f%%'
                        % (fig_names[num], this_bacc * 100,
                           this_gain * 100),
                        xy=(0, .5),
                        xytext=(-2, 0),
                        textcoords='offset points',
                        fontsize=12,
                        va='center', ha='left',
                        xycoords='axes fraction')
        if num == 0:
            dict_coords = {'xy': (0.75, 1), 'xytext': (0, 22),
                           'textcoords': 'offset points',
                           'va': 'center', 'ha': 'center',
                           'xycoords': 'axes fraction',
                           'bbox': {'facecolor': 'black',
                                    'boxstyle': 'round',
                                    'linewidth': 0},
                           'color': 'white',
                           'fontsize': 14}
            ax_full.annotate('Voxelwise decoder', **dict_coords)
            ax_factored.annotate('Task-network decoder\n with multi-study '
                                 'training', **dict_coords)
            dict_coords['xy'] = (0.5, 1.)
            dict_coords['xytext'] = (-5, 22)
            ax_ann.annotate('Classification maps', **dict_coords)

        lw = 2
        offset1 = transforms.ScaledTranslation(-8 * lw / 72. / 2., 0,
                                               fig.dpi_scale_trans)
        trans1 = transforms.blended_transform_factory(
            ax_factored.transAxes + offset1, fig.transFigure)
        l1 = matplotlib.lines.Line2D([0, 0], [0, 1], transform=trans1,
                                     figure=fig, color="black", linewidth=4,
                                     zorder=0)

        offset2 = transforms.ScaledTranslation(-8 * lw / 72. / 2., 0,
                                               fig.dpi_scale_trans)
        trans2 = transforms.blended_transform_factory(
            ax_ann.transAxes + offset2, fig.transFigure)
        l2 = matplotlib.lines.Line2D([0, 0], [0, 1], transform=trans2,
                                     figure=fig, color="black", linewidth=4,
                                     zorder=0)
        fig.lines.extend([l1, l2])

    fig.savefig(join(output_dir, 'classifs.svg'), facecolor=None,
                edgecolor=None,
                transparent=True)
    fig.savefig(join(output_dir, 'classifs.pdf'))


def plot_projection_selection():
    gs = gridspec.GridSpec(3, 3, width_ratios=[2, 2, 1.23],
                           hspace=0., wspace=0.04)
    fig = plt.figure(figsize=(9, 6 * 3 / 5))

    fig.subplots_adjust(left=0.0, right=1., top=1 - 0.085 * 5 / 3,
                        bottom=0.01 * 5 / 3)

    classifs = [
        ('amalric2012mathematicians', 'visual_calculation_vs_baseline'),
        ('ds001', 'pumps_demean_vs_ctrl_demean'),
        ('brainomics', 'visual_processing'),
    ]

    fig_names = [
        "Visual calculation\nAmalaric et al., '12",
        "Pumps vs control\nSchonberg et al., '12",
        "Visual processing\nPapadopoulos et al., '15"
    ]

    output_dir = join(get_output_dir(), 'figure_4')
    names = load(join(output_dir, 'projections', 'names.pkl'))
    index = []
    for name in names:
        study, contrast = name.split('::')
        index.append((study, contrast))
    index = pd.MultiIndex.from_tuples(index)
    indices = pd.Series(data=np.arange(len(names), dtype='int'),
                        index=index, name='index')
    selected = indices.loc[classifs]

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

    classifs_factored = check_niimg(join(output_dir, 'projections',
                                         'components_second.nii.gz'))
    classifs_factored = index_img(classifs_factored,
                                  selected.values.tolist())
    classifs_full = check_niimg(join(output_dir, 'projections',
                                     'components_raw.nii.gz'))
    classifs_full = index_img(classifs_full, selected.values.tolist())
    fig.savefig(join(output_dir, 'figure_4.svg'))
    offset_ann = [6, 0, 0, 4, 0]
    for num, ((study, contrast), index) in enumerate(selected.iteritems()):
        sub_gs_full = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                       subplot_spec=gs[
                                                           num, 0],
                                                       wspace=0.02,
                                                       width_ratios=[2, 1],
                                                       hspace=0.0)
        sub_gs_factored = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                           subplot_spec=
                                                           gs[num, 1],
                                                           wspace=0.02,
                                                           width_ratios=[2,
                                                                         1],
                                                           hspace=0.0)
        ax_full = fig.add_subplot(sub_gs_full[0])
        ax_full_glass = fig.add_subplot(sub_gs_full[1])
        ax_factored = fig.add_subplot(sub_gs_factored[0])
        ax_factored_glass = fig.add_subplot(sub_gs_factored[1])
        ax_ann = fig.add_subplot(gs[num, 2])

        full_img = index_img(classifs_full, num)
        factored_img = index_img(classifs_factored, num)
        vmax = np.abs(factored_img.get_data()).max()
        vmax_full = np.abs(full_img.get_data()).max()

        display_mode = 'xz'

        if contrast == 'visual_calculation_vs_baseline':
            cut_coords = -26, -60, 56  # IPS
        else:
            cut_coords = find_xyz_cut_coords(factored_img,
                                             activation_threshold=vmax / 3)
        s_cut_coords = (cut_coords[0], cut_coords[2])
        ann = 'x = %i, z = %i' % s_cut_coords

        plot_stat_map(full_img, figure=fig,
                      threshold=0,
                      colorbar=False,
                      annotate=False,
                      cut_coords=s_cut_coords,
                      display_mode=display_mode,
                      vmax=vmax_full,
                      axes=ax_full)
        plot_glass_brain(full_img, figure=fig,
                         threshold=vmax_full / 3,
                         plot_abs=False,
                         vmax=vmax_full,
                         cut_coords=cut_coords,
                         colorbar=False,
                         annotate=False,
                         display_mode='z',
                         axes=ax_full_glass)
        plot_stat_map(factored_img, figure=fig,
                      threshold=0,
                      colorbar=False,
                      annotate=False,
                      display_mode=display_mode,
                      vmax=vmax,
                      cut_coords=s_cut_coords,
                      axes=ax_factored)
        plot_glass_brain(factored_img, figure=fig,
                         threshold=vmax / 3,
                         vmax=vmax,
                         colorbar=False,
                         cut_coords=cut_coords,
                         plot_abs=False,
                         display_mode='z',
                         axes=ax_factored_glass)
        ax_full.annotate(ann, xycoords='axes fraction',
                         va='bottom',
                         xytext=(0., offset_ann[num]),
                         textcoords='offset points',
                         bbox=dict(facecolor='white', edgecolor=None,
                                   linewidth=0, pad=0),
                         ha='left', xy=(.03, -0.04))

        ax_ann.axis('off')
        this_gain = gain.loc[(study, contrast)]
        this_bacc = bacc.loc[(study, contrast)]
        ax_ann.annotate('%s\n'
                        'B-accuracy: %.1f%%\nB-acc. gain: %+.1f%%'
                        % (fig_names[num], this_bacc * 100,
                           this_gain * 100),
                        xy=(0, .5),
                        xytext=(-2, 0),
                        textcoords='offset points',
                        fontsize=12,
                        va='center', ha='left',
                        xycoords='axes fraction')
        if num == 0:
            dict_coords = {'xy': (0.75, 1), 'xytext': (0, 18),
                           'textcoords': 'offset points',
                           'va': 'center', 'ha': 'center',
                           'xycoords': 'axes fraction',
                           'bbox': {'facecolor': 'black',
                                    'boxstyle': 'round',
                                    'linewidth': 0},
                           'color': 'white',
                           'fontsize': 14}
            ax_full.annotate('Input z-map', **dict_coords)
            ax_factored.annotate('Projection\n'
                                 'on task-networks',
                                 **dict_coords)
            dict_coords['xy'] = (0.5, 1.)
            dict_coords['xytext'] = (-5, 18)
            ax_ann.annotate('Z-map label', **dict_coords)

        lw = 2
        offset1 = transforms.ScaledTranslation(-8 * lw / 72. / 2., 0,
                                               fig.dpi_scale_trans)
        trans1 = transforms.blended_transform_factory(
            ax_factored.transAxes + offset1, fig.transFigure)
        l1 = matplotlib.lines.Line2D([0, 0], [0, 1], transform=trans1,
                                     figure=fig, color="black",
                                     linewidth=4,
                                     zorder=0)

        offset2 = transforms.ScaledTranslation(-8 * lw / 72. / 2., 0,
                                               fig.dpi_scale_trans)
        trans2 = transforms.blended_transform_factory(
            ax_ann.transAxes + offset2, fig.transFigure)
        l2 = matplotlib.lines.Line2D([0, 0], [0, 1], transform=trans2,
                                     figure=fig, color="black",
                                     linewidth=4,
                                     zorder=0)
        fig.lines.extend([l1, l2])

    fig.savefig(join(output_dir, 'projections.svg'), facecolor=None,
                edgecolor=None,
                transparent=True)
    fig.savefig(join(output_dir, 'projections.pdf'))


plot_classifs_selection()
plot_projection_selection()

# projections()
# classifs_maps()
