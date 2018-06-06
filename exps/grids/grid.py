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
    )
    model = dict(
        estimator='factored',
        normalize=False,
    )
    factored = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        epoch_counting='all',
        sampling='random',
        weight_power=0.6,
        adaptive_dropout=True,
        batch_size=128,
        init='symmetric',
        dropout=0.75,
        lr=1e-3,
        input_dropout=0.25,
        seed=100,
        max_iter={'pretrain': 300, 'sparsify': 0, 'finetune': 200},
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
        studies='all'
    )
    model = dict(
        normalize=False,
        estimator='logistic',
    )
    logistic = dict(
        max_iter=1000,
        solver='lbfgs',
        l2_penalty=1e-4,
    )


def full_logistic():
    seed = 1
    system = dict(
        device=-1,
        verbose=100,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'masked'),
        studies='all'
    )
    model = dict(
        normalize=False,
        estimator='logistic',
    )
    logistic = dict(
        l2_penalty=1e-6,
        solver='saga',
        max_iter=1000
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
    )
    model = dict(
        normalize=False,
        estimator='factored',
    )
    factored_variational = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        epoch_counting='all',
        adaptive_dropout=False,
        sampling='random',
        max_iter={'pretrain': 300, 'sparsify': 0, 'finetune': 200},
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

    source_dir = join(get_data_dir(), 'reduced_512')
    _, target = load_data_from_dir(data_dir=source_dir)
    studies = list(target.keys())
    seeds = check_random_state(42).randint(0, 100000, size=20)

    if grid == 'seed_split_init':
        output_dir = join(get_output_dir(), 'seed_split_init')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(factored)
        model_seeds = check_random_state(43).randint(0, 100000, size=100)
        config_updates = ParameterGrid({'factored.seed': model_seeds,
                                        'seed': seeds})
    if grid == 'weight_power':
        output_dir = join(get_output_dir(), 'weight_power')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(factored)
        weight_power = np.linspace(0, 1, 10)
        config_updates = ParameterGrid({'factored.weight_power': weight_power,
                                        'seed': seeds})

    elif grid == 'dropout':
        output_dir = join(get_output_dir(), 'dropout')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(factored)
        dropout = [0.5, 0.75]
        adaptive_dropout = [False, True]
        config_updates = ParameterGrid({'seed': seeds[:10],
                                        'factored.dropout': dropout,
                                        'factored.adaptive_dropout': adaptive_dropout})
    elif grid in ['reduced_logistic', 'full_logistic']:
        if grid == 'reduced_logistic':
            exp.config(reduced_logistic)
            l2_penalties = np.logspace(-4, 0, 5)

            output_dir = join(get_output_dir(), 'reduced_logistic')
        elif grid == 'full_logistic':
            exp.config(full_logistic)
            l2_penalties = [1e-3]
            output_dir = join(get_output_dir(), 'full_logistic')
            seeds = seeds[:3]
        config_updates = ParameterGrid({'logistic.l2_penalty': l2_penalties,
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
