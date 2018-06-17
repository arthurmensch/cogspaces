import numpy as np
from cytoolz import join
from joblib import dump
from os.path import join

from cogspaces.datasets.utils import get_output_dir, get_data_dir
from cogspaces.utils import get_dictionary
from exps.analyse.plot_maps import get_components
from exps.train import load_data


def compute_projections(layer='first', output_dir=None):
    data, target = load_data(join(get_data_dir(), 'reduced_512_gm'),
                             'all')

    first_proj = get_dictionary()
    if layer == 'first':
        full_proj = first_proj
        output_dir = join(get_output_dir(), 'projected_512_gm')
    else:
        assert layer == 'second'
        second_proj = get_components(output_dir, return_type='arrays')
        full_proj = second_proj.dot(first_proj)

    for study, this_data in data.items():
        this_target = target[study]
        gram = full_proj.dot(full_proj.T)
        if layer == 'first':
            # already projected
            y = this_data
            filename = join(output_dir, 'data_%s.pt' % study)
        else:
            y = this_data.dot(second_proj.T)
            filename = join(output_dir, 'projected', 'data_%s.pt' % study)
        data_rec = y.dot(np.linalg.inv(gram)).dot(full_proj)
        dump((data_rec, this_target), filename)


compute_projections('first')
compute_projections('second', join(get_output_dir(),
                                   'factored_gm_full', '1'))
