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
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-5,
                  'finetune': 1e-3},
        batch_size=128,
        max_iter={'pretrain': 10, 'train': 10,
                  'sparsify': 0, 'finetune': 10},
        seed=100,
    )


def factored_single():
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
        adaptive_dropout=False,
        batch_norm=True,
        regularization=1,
        input_dropout=0.25,
        dropout=0.75,
        optimizer='adam',
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-3,
                  'finetune': 1e-3},
        batch_size=128,
        max_iter={'pretrain': 200, 'train': 300,
                  'sparsify': 0, 'finetune': 200},
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
        lr={'pretrain': 1e-3, 'train': 1e-4, 'sparsify': 1e-3,
                  'finetune': 1e-3},
        batch_size=128,
        max_iter={'pretrain': 200, 'train': 100,
                  'sparsify': 0, 'finetune': 200},
        seed=100)


def factored_logistic_refit():
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


def logistic():
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
        l2_penalty=np.logspace(-5, 1, 7).tolist(),
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


def study_selector():
    seed = 10
    full = False
    system = dict(
        device=-1,
        verbose=2,
        n_jobs=1,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='all',
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
        study_selector=True,
        target_study='brainomics',
    )
    factored = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        adaptive_dropout=True,
        sampling='random',
        weight_power=0.6,
        batch_size=128,
        init='rest',
        batch_norm=True,
        # full_init=join(get_output_dir(), 'seed_split_init', 'pca_15795.pkl'),
        dropout=0.75,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4,
                  'finetune': 1e-3},
        input_dropout=0.25,
        max_iter={'pretrain': 200, 'train': 300, 'sparsify': 0,
                  'finetune': 200},
    )

    logistic = dict(
        estimator='logistic',
        l2_penalty=np.logspace(-5, 1, 7).tolist(),
        max_iter=1000,
        reduction=None
    )


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
    model_seeds = check_random_state(143).randint(0, 1000000, size=2)

    output_dir = join(get_output_dir(), grid)

    if grid == 'factored_dense_init':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.init': 'orthogonal',
                                        'factored.seed': model_seeds,
                                        })
    elif grid == 'factored_sparsify_less':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.max_iter.sparsify': [200],
                                        })
    elif grid == 'factored':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.seed': model_seeds,
                                        })
    elif grid == 'factored_study_selector':
        exp.config(study_selector)
        config_updates = ParameterGrid({'model.target_study': studies,
                                        'factored.seed': model_seeds[:5],
                                        })
    elif grid == 'factored_refit_cautious':
        exp.config(factored_refit)
        init_dir = join(get_output_dir(), 'factored')

        config_updates = [{'seed': seed,
                           'factored.full_init': join(init_dir,
                                                      '%s_%i.pkl' %
                                                      (decomposition, seed))}
                          for seed in seeds
                          for decomposition in ['dl_rest', 'dl_random']]
    elif grid == 'full':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': [0],
                                        'full': [True],
                                        'factored.seed': model_seeds,
                                        })

    elif grid == 'factored_logistic_refit':
        exp.config(factored_logistic_refit)
        seed_split_init_dir = join(get_output_dir(), 'factored')
        l2_penalties = np.logspace(-4, 0, 5)

        config_updates = [{'seed': seed,
                           'logistic.reduction': join(seed_split_init_dir,
                                                      '%s_%i.npy' %
                                                      (init, seed)),
                           'logistic.l2_penalty': l2_penalty}
                          for seed in seeds
                          for init in ['pca', 'dl_rest',
                                       'dl_random']
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
    elif grid in ['logistic', 'full_logistic']:
        if grid == 'logistic':
            exp.config(logistic)
        elif grid == 'full_logistic':
            exp.config(full_logistic)
        config_updates = ParameterGrid({'data.studies': studies,
                                        'seed': seeds})
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
