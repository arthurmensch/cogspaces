import sys

import numpy as np
import os
import pandas as pd
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


def factored_pretrain():
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
        normalize=False)
    factored = dict(
        weight_power=0.6,
        latent_size=128,
        activation='linear',
        epoch_counting='all',
        sampling='random',
        init='rest',
        adaptive_dropout=True,
        batch_norm=True,
        regularization=1,
        input_dropout=0.25,
        dropout=0.75,
        optimizer='adam',
        lr=1e-3,
        batch_size=128,
        max_iter={'pretrain': 200, 'train': 300,
                  'sparsify': 200, 'finetune': 200},
        seed=100,
    )


def factored_refit():
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
        batch_norm=False,
        full_init=None,
        lr=1e-3,
        input_dropout=0.25,
        seed=100,
        max_iter={'pretrain': 0, 'sparsify': 0, 'finetune': 500},
    )


def logistic_refit():
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
        estimator='logistic',
        normalize=False,
    )
    logistic = dict(
        max_iter=4000,
        solver='lbfgs',
        l2_penalty=1e-4,
        reduction=None,
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
        max_iter=2000,
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

    output_dir = join(get_output_dir(), grid)

    if grid == 'seed_split_init':
        exp.config(factored)
        model_seeds = check_random_state(143).randint(100000, 1000000,
                                                      size=200)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.seed': model_seeds,
                                        })
    elif grid == 'factored_pretrain':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.max_iter.sparsify': [0, 200],
                                        })
    elif grid == 'full':
        exp.config(factored)
        model_seeds = check_random_state(143).randint(100000, 1000000,
                                                      size=200)
        config_updates = ParameterGrid({'seed': [0],
                                        'full': [True],
                                        'factored.seed': model_seeds,
                                        })
    elif grid == 'init_refit':
        exp.config(factored_refit)
        seed_split_init_dir = join(get_output_dir(), 'seed_split_init')

        finetune_dropouts = pd.read_pickle(join(seed_split_init_dir,
                                                'dropout.pkl'))
        config_updates = [{'seed': seed,
                           'factored.full_init': join(seed_split_init_dir,
                                                 '%s_%i.npy' %
                                                 (decomposition, seed))}
                          for seed in seeds
                          for decomposition in ['pca', 'dl_rest', 'dl_random']]
    elif grid == 'logistic_refit_l2':
        exp.config(logistic_refit)
        seed_split_init_dir = join(get_output_dir(), 'seed_split_init')
        l2_penalties = np.logspace(-4, 0, 5)

        config_updates = [{'seed': seed,
                           'logistic.reduction': join(seed_split_init_dir,
                                                      '%s_%i.npy' %
                                                      (init, seed)),
                           'logistic.l2_penalty': l2_penalty}
                          for seed in seeds
                          for init in ['pca', 'dl_rest_init',
                                       'dl_random_init']
                          for l2_penalty in l2_penalties]
    elif grid == 'single_factored':
        exp.config(factored)
        config_updates = ParameterGrid({'data.studies': studies,
                                        'seed': seeds})
    elif grid == 'weight_power':
        exp.config(factored)
        weight_power = np.linspace(0, 1, 10)
        config_updates = ParameterGrid({'factored.weight_power': weight_power,
                                        'seed': seeds})

    elif grid == 'dropout':
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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError('Directory exists.')

    _id = get_id(output_dir)
    Parallel(n_jobs=40, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      mock=False,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))
