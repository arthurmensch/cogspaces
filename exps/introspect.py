import os
from os.path import join

import matplotlib.pyplot as plt
from joblib import load, dump
from nilearn.image import index_img
from nilearn.plotting import plot_stat_map

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import get_output_dir
from cogspaces.introspect.maps import maps_from_model


def plot_components(components, names, output_dir):
    for study in components:
        this_img = components[study]
        these_names = names[study]

        for i, name in enumerate(these_names):
            full_name = study + ' ' + name
            fig = plt.figure()
            plot_stat_map(index_img(this_img, i), figure=fig, title=full_name)
            plt.savefig(join(output_dir, '%s.png' % full_name))
            plt.close(fig)


def compute_components(output_dir, lstsq):
    estimator = load(join(output_dir, 'estimator.pkl'))
    target_encoder = load(join(output_dir, 'target_encoder.pkl'))
    standard_scaler = load(join(output_dir, 'standard_scaler.pkl'))

    modl_atlas = fetch_atlas_modl()
    dictionary = modl_atlas['components512']
    components, names = maps_from_model(estimator, dictionary,
                                        target_encoder,
                                        standard_scaler,
                                        lstsq=lstsq)
    plot_dir = join(output_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    dump(names, join(plot_dir, 'names.pkl'))
    for study, this_components in components.items():
        this_components.to_filename(join(plot_dir,
                                         'components_%s.nii.gz' % study))
        plot_components(components, names, plot_dir)


if __name__ == '__main__':
    # compute_components(join(get_output_dir(), 'multi_studies', '57'))
    compute_components(join(get_output_dir(), 'multi_studies', '107'),
                       lstsq=True)
