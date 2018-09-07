import matplotlib
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

mpl.use('pgf')
mplparams = {
    "font.family": "sans-serif",
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
}
mpl.rcParams.update(mplparams)
# mpl.rcParams['font.family'] = 'cmss10'

import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
from jinja2 import Template
from joblib import delayed, Parallel, load, Memory
from matplotlib import transforms
from matplotlib.testing.compare import get_cache_dir
from nibabel.tests.test_viewers import matplotlib
from nilearn._utils import check_niimg
from nilearn.image import iter_img, index_img
from nilearn.plotting import find_xyz_cut_coords, plot_stat_map, \
    plot_glass_brain
from os.path import join

from cogspaces.datasets.utils import get_output_dir

import matplotlib.pyplot as plt


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


def plot_classifs_selection(mode='direct'):
    mem = Memory(cachedir=get_cache_dir())
    gs = gridspec.GridSpec(5, 3, width_ratios=[1.45, 2, 2],
                           hspace=0., wspace=0.04)
    fig = plt.figure(figsize=(9, 6.5))

    fig.subplots_adjust(left=0.0, right=1., top=.89, bottom=0.01)

    classifs = [('ds105', 'face_vs_house'),
                ('ds001', 'pumps_demean_vs_ctrl_demean'),
                ('pinel2009twins', 'language_vs_sound'),
                ('cauvet2009muslang', 'c16_c08_vs_c01_C02_music'),
                ('brainomics', 'vertical_checkerboard')
                ]

    # row_labels = ["Face vs house\nHaxby et al.$^{60}$",
    #               "Pumps vs control\nSchonberg et al.$^{70}$",
    #               "Language vs sound\nPinel et al.$^{66}$",
    #               "Complex vs simple music\nCauvet et al.$^{50}$",
    #               "Vertical checkerboard\nPapadopoulos O. et al.$^{65}$"
    #               ]
    row_labels = ["Face vs house\nHaxby \\textit{et al.}\\textsuperscript{60}",
                  "Pumps vs control\nSchonberg \\textit{et al.}\\textsuperscript{70}",
                  "Language vs sound\nPinel \\textit{et al.}\\textsuperscript{66}",
                  "Complex vs simple music\nCauvet \\textit{et al.}\\textsuperscript{50}",
                  "Vertical checkerboard\nPapadopoulos O. \\textit{et al.}\\textsuperscript{65}"
                  ]

    ann_offsets = [5, 0, 2, 4, 10]

    cut_coords = [[25.41189480229025, -39.77450008662751, -12.81076684554079],
                  # auto
                  [-44.208589241705766, -20.388810005566825,
                   53.50303011685118],
                  [-54, 0, -8],
                  [-52, 0, 42],
                  [-2.705459506699043, -96.00920144425464, 16.526994436628627]
                  ]
    display_modes = ['xz', 'xz', 'xz', 'xz', 'yz']

    label_props_wb = {'xytext': (0, 22),
                   'textcoords': 'offset points',
                   'va': 'center', 'ha': 'center',
                   'xycoords': 'axes fraction',
                   'bbox': {'facecolor': 'black',
                            'boxstyle': 'round',
                            'linewidth': 0},
                   'color': 'white',
                   'fontsize': 17}
    label_props = {'xytext': (0, 22),
                   'textcoords': 'offset points',
                   'va': 'center', 'ha': 'center',
                   'xycoords': 'axes fraction',
                   'fontsize': 17}

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

    bacc, gain = mem.cache(get_bacc_gain)()

    classifs_factored, classifs_full, proj_factored, proj_full = mem.cache(
        get_imgs)(
        output_dir, selected.values.tolist())

    if mode == 'direct':
        imgs_list = [classifs_full, classifs_factored]
        col_labels = ['Contrast label',
                      'Voxelwise decoder',
                      'Task-network decoder',
                      ]
    else:
        imgs_list = [proj_full, proj_factored]
        col_labels = ['Contrast label',
                      'Raw z-map',
                      'Projection on task networks'
                      ]

    for row, ((study, contrast), index) in enumerate(selected.iteritems()):
        for column, imgs in enumerate(imgs_list):
            sub_gs = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                      subplot_spec=gs[
                                                          row, column + 1],
                                                      wspace=0.02,
                                                      width_ratios=[2, 1],
                                                      hspace=0.0)
            ax_stat = fig.add_subplot(sub_gs[0])
            ax_glass = fig.add_subplot(sub_gs[1])

            img = index_img(imgs, row)
            vmax = np.abs(img.get_data()).max()
            cc = cut_coords[row]
            display_mode = display_modes[row]
            if display_mode == 'xz':
                rcc = (cc[0], cc[2])
                ann = 'x = %i, z = %i' % rcc
            elif display_mode == 'yz':
                rcc = (cc[1], cc[2])
                ann = 'y = %i, z = %i' % rcc

            plot_stat_map(img, figure=fig,
                          threshold=0,
                          colorbar=False,
                          annotate=False,
                          cut_coords=rcc,
                          display_mode=display_mode,
                          vmax=vmax,
                          axes=ax_stat)
            plot_glass_brain(img, figure=fig,
                             threshold=vmax / 2.5 if contrast == 'language_vs_sound' else vmax / 3,
                             plot_abs=False,
                             vmax=vmax,
                             cut_coords=cut_coords,
                             colorbar=False,
                             annotate=column == 1,
                             display_mode='z',
                             axes=ax_glass)

            if column == 0:
                ax_stat.annotate(ann, xycoords='axes fraction',
                                 va='bottom',
                                 xytext=(0., ann_offsets[row]),
                                 fontsize=10,
                                 textcoords='offset points',
                                 bbox=dict(facecolor='white', edgecolor=None,
                                           linewidth=0, pad=0),
                                 ha='left', xy=(.03, -0.04))

            if row == 0:
                if column == 0:
                    label = 'Classification maps' if mode == 'direct' else 'Input data transformation'
                    font = FontProperties(weight='bold')
                    ax_glass.annotate(label, xy=(1., 1.19), zorder=1,
                                      fontproperties=font,
                                      **label_props_wb)

                ax_stat.annotate(col_labels[column], xy=(0.75, 0.91),
                                 **label_props)
                lw = 2
                offset = transforms.ScaledTranslation(-8 * lw / 72. / 2., 0,
                                                      fig.dpi_scale_trans)
                trans = transforms.blended_transform_factory(
                    ax_stat.transAxes + offset, fig.transFigure, )
                l = matplotlib.lines.Line2D([0, 0], [0, 1] if column == 0 else [0, .95],
                                            transform=trans,
                                            figure=fig, color='black',
                                            linewidth=2,
                                            zorder=0)
                fig.lines.append(l)
        # if row == 0:
        #     trans = transforms.blended_transform_factory(
        #         fig.transFigure, ax_stat.transAxes + offset)
        #     l = matplotlib.lines.Line2D([1, 0], [1, 1],
        #                                 transform=trans,
        #                                 figure=fig, color='black',
        #                                 linewidth=2,
        #                                 zorder=0)
        fig.lines.append(l)
        ax_ann = fig.add_subplot(gs[row, 0])
        ax_ann.axis('off')
        this_gain = gain.loc[(study, contrast)]
        this_bacc = bacc.loc[(study, contrast)]
        ax_ann.annotate('%s\n'
                        'B-accuracy: %.1f%%\nB-acc. gain: %+.1f%%'
                        % (row_labels[row], this_bacc * 100,
                           this_gain * 100),
                        xy=(1, .5),
                        xytext=(-5, 0),
                        textcoords='offset points',
                        fontsize=15,
                        va='center', ha='right',
                        xycoords='axes fraction')
        if row == 0:
            ax_ann.annotate(col_labels[0], xy=(0.5, 0.9),
                            **label_props)

    # fig.savefig(join(output_dir, 'classifs_%s.svg' % mode),
    #             transparent=True)
    fig.savefig(join(output_dir, 'classifs_%s.pdf' % mode))


def get_imgs(output_dir, selected):
    classifs_factored = check_niimg(join(output_dir, 'classifs',
                                         'classifs_factored.nii.gz'))
    classifs_factored = index_img(classifs_factored, selected)
    classifs_full = check_niimg(join(output_dir, 'classifs',
                                     'classifs_full.nii.gz'))
    classifs_full = index_img(classifs_full, selected)
    proj_factored = check_niimg(join(output_dir, 'projections',
                                     'components_second.nii.gz'))
    proj_factored = index_img(proj_factored, selected)
    proj_full = check_niimg(join(output_dir, 'projections',
                                 'components_raw.nii.gz'))
    proj_full = index_img(proj_full, selected)
    return classifs_factored, classifs_full, proj_factored, proj_full


def get_bacc_gain():
    metrics = pd.read_pickle(join(get_output_dir(), 'figure_4',
                                  'metrics', 'metrics.pkl'))
    baseline = pd.read_pickle(join(get_output_dir(), 'figure_4',
                                   'metrics', 'baseline', 'metrics.pkl'))
    metrics = metrics.loc[1e-4]
    bacc = metrics['bacc'].groupby(['study', 'contrast']).mean()
    gain = (metrics['bacc'] - baseline['bacc']).groupby(
        ['study', 'contrast']).mean()
    return bacc, gain


plot_classifs_selection('direct')
plot_classifs_selection('proj')
# plot_projection_selection()

# projections()
# classifs_maps()
