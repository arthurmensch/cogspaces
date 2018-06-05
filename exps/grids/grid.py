import sys

import numpy as np
import os
from joblib import Parallel, delayed
from os.path import join
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id, OurFileStorageObserver
from exps.train import exp


def factored():
    seed = 10
    full = False
    system = dict(
        device=-1,
        verbose=2,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='all',
        target_study='archi',
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
        max_iter={'pretrain': 300, 'sparsify': 0, 'finetune': 200},
    )
    factored = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        epoch_counting='all',
        sampling='random',
        weight_power=0.6,
        batch_size=128,
        init='symmetric',
        dropout=0.5,
        lr=1e-3,
        input_dropout=0.25
    )

    logistic = dict(
        l2_penalty=1e-4,
    )


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
        normalize=False,
        estimator='logistic',
        max_iter={'pretrain': 1000},
    )
    logistic = dict(
        l2_penalty=1e-6,
    )


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
        normalize=False,
        estimator='logistic',
        max_iter={'pretrain': 500},
    )
    logistic = dict(
        l2_penalty=1e-6,
        solver='saga'
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


def run_exp(output_dir, config_updates, _id, mock=False):
    """Boiler plate function that has to be put in every multiple
        experiment script, as exp does not pickle."""
    if not mock:
        observer = OurFileStorageObserver.create(basedir=output_dir)

        run = exp._create_run(config_updates=config_updates, )
        run._id = _id
        run.observers.append(observer)
        try:
            run()
        except:
            print('Failed at some point. Continuing')
            return
    else:
        exp.run_command('print_config', config_updates=config_updates, )


if __name__ == '__main__':
    grid = sys.argv[1]

    source_dir = join(get_data_dir(), 'reduced_512_lstsq')
    _, target = load_data_from_dir(data_dir=source_dir)
    studies = list(target.keys())

    if grid == 'seed_split_init':
        output_dir = join(get_output_dir(), 'seed_split_init')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(factored)
        seeds = check_random_state(42).randint(0, 100000, size=20)
        model_seeds = check_random_state(43).randint(0, 100000, size=100)
        config_updates = ParameterGrid({'model.seed': model_seeds,
                                        'seed': seeds})
    if grid == 'dropout':
        output_dir = join(get_output_dir(), 'dropout')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(factored)
        seeds = check_random_state(42).randint(0, 100000, size=20)
        dropout = [0.5, 0.75]
        adaptive = [False, True]
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.dropout': seeds})
    elif grid in ['reduced_logistic', 'full_logistic']:
        if grid == 'reduced_logistic':
            exp.config(reduced_logistic)
            l2_penalties = np.logspace(-4, 0, 5)

            output_dir = join(get_output_dir(), 'reduced_logistic')
            seeds = check_random_state(42).randint(0, 100000, size=20)
        elif grid == 'full_logistic':
            exp.config(full_logistic)
            l2_penalties = [1e-3]
            output_dir = join(get_output_dir(), 'full_logistic')
            seeds = check_random_state(42).randint(0, 100000, size=1)
            seeds = seeds[:3]
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
        raise ValueError('Wrong argument')

    _id = get_id(output_dir)
    Parallel(n_jobs=40, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      mock=False,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))
