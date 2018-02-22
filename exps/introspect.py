from os.path import join

import os

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import get_output_dir
from cogspaces.introspect.maps import maps_from_model, plot_components
from joblib import load, dump


def plot_output(output_dir):
    estimator = load(join(output_dir, 'estimator.pkl'))
    target_encoder = load(join(output_dir, 'target_encoder.pkl'))
    standard_scaler = load(join(output_dir, 'standard_scaler.pkl'))

    modl_atlas = fetch_atlas_modl()
    dictionary = modl_atlas['components512']
    components, names = maps_from_model(estimator, dictionary,
                                        target_encoder,
                                        standard_scaler)
    plot_dir = join(output_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    dump(components, join(plot_dir, 'components.pkl'))
    dump(names, join(plot_dir, 'names.pkl'))
    plot_components(components, names, plot_dir)


plot_output(join(get_output_dir(), 'multi_studies', '53'))
