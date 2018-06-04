import sys

import numpy as np
import pandas as pd
from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id
from exps.grids.grid import run_exp
from exps.train import exp
from joblib import Parallel, delayed
from os.path import join
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state


@exp.config
def reduced_logistic():
    seed = 1
    system = dict(
        device=-1,
        verbose=100,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='archi'
    )
    model = dict(
        normalize=True,
        estimator='logistic',
        max_iter=200,
    )
    logistic = dict(
        l2_penalty=1e-6,
    )


@exp.config
def full_logistic():
    seed = 1
    system = dict(
        device=-1,
        verbose=100,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'masked'),
        studies='archi'
    )
    model = dict(
        normalize=True,
        estimator='logistic',
        max_iter=200,
    )
    logistic = dict(
        l2_penalty=1e-6,
    )


def reduced_factored():
    seed = 10
    full = False
    system = dict(
        device=-1,
        verbose=0,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='all',
        target_study='archi'
    )
    model = dict(
        normalize=False,
        estimator='factored_variational',
        study_weight='sqrt_sample',
        max_iter={'pretrain': 300, 'sparsify': 0, 'finetune': 200},
    )
    factored_variational = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        epoch_counting='all',
        sampling='random',
        weight_power=0.6,
        batch_size=128,
        dropout=0.5,
        lr=1e-3,
        input_dropout=0.25)


def main():
    grid = sys.argv[1]
    _, target = load_data_from_dir(data_dir=join(get_data_dir(),
                                                 'reduced_512'))
    studies = list(target.keys())

    if grid in ['reduced_logistic', 'full_logistic']:
        if grid == 'reduced_logistic':
            exp.config(reduced_logistic)
            output_dir = join(get_output_dir(), 'reduced_logistic')
        elif grid == 'full_logistic':
            exp.config(full_logistic)
            output_dir = join(get_output_dir(), 'full_logistic')
        seeds = check_random_state(42).randint(0, 100000, size=20)
        l2_penalties = np.logspace(-4, 0, 20)
        config_updates = ParameterGrid(
            {'logistic.l2_penalty': l2_penalties,
             'data.studies': studies,
             'seed': seeds})
    elif grid == 'reduced_factored':
        exp.config(reduced_factored)
        seeds = check_random_state(42).randint(0, 100000, size=20)
        config_updates = ParameterGrid({'data.studies': studies,
                                        'seed': seeds})
        output_dir = join(get_output_dir(), 'reduced_logistic')
    else:
        raise ValueError('Wrong grid name %s' % grid)

    _id = get_id(output_dir)
    Parallel(n_jobs=20, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))


def refit():
    baseline_res = pd.read_pickle(join(get_output_dir(), 'baseline_avg.pkl'))

    studies = baseline_res.index.get_level_values('study').tolist()
    l2_penalties = baseline_res[('l2_penalty_', 'mean')].values.tolist()
    config_updates = [
        {'data.studies': study, 'logistic.l2_penalty': l2_penalty,
         'full': True
         } for study, l2_penalty in zip(studies, l2_penalties)]

    output_dir = join(get_output_dir(), 'baseline_logistic_refit')

    _id = get_id(output_dir)

    Parallel(n_jobs=20, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))


if __name__ == '__main__':
    main()
