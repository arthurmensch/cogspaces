import os
import sys
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id, OurFileStorageObserver
from exps.grids.gather_quantitative import get_studies
from exps.train import exp


def factored():
    seed = 100
    full = False
    system = dict(
        device=-1,
        verbose=2,
        n_jobs=1,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_gm'),
        studies='all',
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
        refinement=None,
        target_study=None,
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
        epoch_counting='all',
        init='rest_gm',
        batch_norm=True,
        dropout=0.75,
        input_dropout=0.25,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4,
            'finetune': 1e-3},
        max_iter={'pretrain': 200, 'train': 300, 'sparsify': 0,
                  'finetune': 200},
    )


def training_curve():
    studies = get_studies()
    data = dict(train_size={study: 1. for study in studies},
                test_size={study: 0. for study in studies})


def trace():
    seed = 860
    full = False
    system = dict(
        device=-1,
        verbose=5,
        n_jobs=3,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_gm'),
        studies=['archi', 'hcp'],
    )
    model = dict(
        estimator='trace',
        normalize=True,
        seed=100,
    )

    trace = dict(
        trace_penalty=1e-1,
        max_iter=6000,
        momentum=True,
    )
    factored = dict(
        optimizer='adam',
        latent_size='auto',
        activation='linear',
        regularization=1,
        adaptive_dropout=True,
        sampling='random',
        weight_power=0.6,
        batch_size=128,
        epoch_counting='all',
        init='rest_gm',
        batch_norm=True,
        dropout=0.75,
        input_dropout=0.25,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4,
            'finetune': 1e-3},
        max_iter={'pretrain': 200, 'train': 300, 'sparsify': 0,
                  'finetune': 200},
    )


def factored_transfer():
    seed = 100
    full = False
    system = dict(
        device=-1,
        verbose=2,
        n_jobs=1,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_gm'),
        studies='all',
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
        refinement=None,
        target_study=None,
    )
    factored = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        adaptive_dropout=True,
        black_list_target=True,
        sampling='random',
        weight_power=0.6,
        batch_size=128,
        epoch_counting='all',
        init='normal',
        batch_norm=True,
        reset_classifiers=False,
        dropout=0.5,
        input_dropout=0.25,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4,
            'finetune': 1e-3},
        max_iter={'pretrain': 0, 'train': 300, 'sparsify': 0,
                  'finetune': 200},
    )


def factored_single():
    seed = 100
    full = False
    system = dict(
        device=-1,
        verbose=2,
        n_jobs=1,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_gm'),
        studies='all',
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
        refinement=None,
        target_study=None,
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
        epoch_counting='all',
        init='rest_gm',
        batch_norm=True,
        dropout=0.5,
        input_dropout=0.25,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4,
            'finetune': 1e-3},
        max_iter={'pretrain': 200, 'train': 300, 'sparsify': 0,
                  'finetune': 200},
    )


def factored_refit():
    seed = 10
    full = False
    system = dict(
        device=-1,
        verbose=2,
        n_jobs=1,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_gm'),
        studies='all',
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
        refinement=None,
        target_study=None,
    )
    factored = dict(
        weight_power=0.6,
        latent_size=128,
        activation='linear',
        epoch_counting='all',
        sampling='random',
        init='rest_gm',
        adaptive_dropout=True,
        black_list_target=False,
        reset_classifiers=False,
        batch_norm=True,
        regularization=1,
        input_dropout=0.25,
        dropout=0.75,
        optimizer='adam',
        lr={'pretrain': 1e-3, 'train': 1e-5, 'sparsify': 1e-3,
            'finetune': 1e-3},
        batch_size=128,
        max_iter={'pretrain': 0, 'train': 300,
                  'sparsify': 0, 'finetune': 200},
        seed=100)


def logistic():
    seed = 1
    system = dict(
        device=-1,
        verbose=0,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_gm'),
        studies='all'
    )
    model = dict(
        normalize=False,
        estimator='logistic',
    )
    logistic = dict(
        estimator='logistic',
        max_iter=2000,
        solver='lbfgs',
        l2_penalty=np.logspace(-5, 1, 7).tolist(),
        refit_from=None

    )


def full_logistic():
    seed = 1
    system = dict(
        device=-1,
        verbose=100,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'masked_gm'),
        studies='all'
    )
    model = dict(
        normalize=False,
        estimator='logistic',
    )
    logistic = dict(
        estimator='logistic',
        l2_penalty=[1e-3],
        solver='lbfgs',
        max_iter=2000
    )


def study_selector():
    seed = 100
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
        refinement='study_selector',
        target_study=None,
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
        epoch_counting='all',
        init='rest',
        batch_norm=True,
        dropout=0.5,
        input_dropout=0.25,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4,
            'finetune': 1e-3},
        max_iter={'pretrain': 200, 'train': 300, 'sparsify': 0,
                  'finetune': 200},
    )

    refinement = dict(
        n_runs=1,
        n_splits=3
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
    model_seeds = check_random_state(243).randint(0, 1000000, size=120)

    output_dir = join(get_output_dir(), grid)

    if grid == 'training_curves':
        exp.config(factored)
        exp.config(training_curve)
        config_updates = []
        for seed in seeds:
            for study in ['brainomics', 'archi', 'henson2010faces', 'camcan']:
                for size in np.linspace(.1, .5, 5):
                    config_updates.append(
                        {'seed': seed,
                         'data.train_size': {study: size},
                         'data.test_size': {study: .5}
                         }
                    )
    elif grid == 'factored_gm':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.seed': model_seeds,
                                        })
    elif grid == 'factored_gm_full':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': [0],
                                        'full': [True],
                                        'factored.seed': model_seeds,
                                        })
    elif grid == 'factored_gm_normal_init':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.seed': model_seeds,
                                        'factored.init': ['normal'],
                                        'factored.max_iter.pretrain': [0]
                                        })
    elif grid == 'reset_classifiers':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.seed': [0],
                                        'factored.init': ['normal'],
                                        'factored.reset_classifiers': [True,
                                                                       False],
                                        'factored.max_iter.pretrain': [0]
                                        })
    elif grid == 'factored_gm_norm_init_full':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': [0],
                                        'factored.seed': model_seeds,
                                        'factored.init': ['normal'],
                                        'full': [True],
                                        'factored.max_iter.pretrain': [0]
                                        })
    elif grid == 'factored_study_selector':
        exp.config(study_selector)
        config_updates = ParameterGrid({'model.target_study': studies,
                                        'seed': seeds,
                                        })
    elif grid == 'factored_refit_gm_lower_lr':
        exp.config(factored_refit)
        init_dir = join(get_output_dir(), 'factored_gm')

        config_updates = [{'seed': seed,
                           'factored.refit_from': join(init_dir,
                                                       '%s_%i_%.0e.pkl' %
                                                       (decomposition, seed,
                                                        alpha)),
                           'factored.refit_data': ['dropout', 'classifier'],
                           }
                          for seed in seeds
                          for alpha in [1e-2, 1e-3, 1e-4]
                          for decomposition in ['dl_rest']]
    elif grid == 'factored_refit_gm_rest_positive_notune':
        exp.config(factored_refit)
        init_dir = join(get_output_dir(), 'factored_gm')

        config_updates = [{'seed': seed,
                           'factored.refit_from': join(init_dir,
                                                       '%s_%i_%.0e.pkl' %
                                                       (decomposition, seed,
                                                        alpha)),
                           'factored.refit_data': ['classifier', 'dropout'],
                           'factored.max_iter': {'pretrain': 0,
                                                 'train': 0,
                                                 'sparsify': 0,
                                                 'finetune': 0}
                           }
                          for seed in seeds
                          for alpha in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                          for decomposition in ['dl_rest_positive']]
    elif grid == 'factored_refit_gm_full_rest_positive_notune':
        exp.config(factored_refit)
        init_dir = join(get_output_dir(), 'factored_gm_full')

        config_updates = [{'seed': 0,
                           'factored.refit_from': join(init_dir,
                                                       '%s_%i_%.0e.pkl' %
                                                       (decomposition, 0,
                                                        alpha)),
                           'factored.refit_data': ['classifier', 'dropout'],
                           'factored.max_iter': {'pretrain': 0,
                                                 'train': 0,
                                                 'sparsify': 0,
                                                 'finetune': 0}
                           }
                          for alpha in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                          for decomposition in ['dl_rest_positive']]

    elif grid == 'factored_refit_gm_full_positive_notune':
        exp.config(factored_refit)
        init_dir = join(get_output_dir(), 'factored_gm_normal_init_full')

        config_updates = [{'seed': 0,
                           'factored.refit_from': join(init_dir,
                                                       '%s_%i_%.0e.pkl' %
                                                       (decomposition, 0,
                                                        alpha)),
                           'factored.refit_data': ['classifier', 'dropout'],
                           'factored.max_iter': {'pretrain': 0,
                                                 'train': 0,
                                                 'sparsify': 0,
                                                 'finetune': 0}
                           }
                          for alpha in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                          for decomposition in ['dl_positive']]

    elif grid == 'factored_refit_gm_notune':
        exp.config(factored_refit)
        init_dir = join(get_output_dir(), 'factored_gm')

        config_updates = [{'seed': seed,
                           'factored.refit_from': join(init_dir,
                                                       '%s_%i_%.0e.pkl' %
                                                       (decomposition, seed,
                                                        alpha)),
                           'factored.refit_data': []}
                          for seed in seeds
                          for alpha in [1e-2, 1e-3, 1e-4]
                          for decomposition in ['dl_rest']]
    elif grid == 'factored_refit_gm_normal_init_low_lr':
        exp.config(factored_refit)
        init_dir = join(get_output_dir(), 'factored_gm_normal_init')
        config_updates = [{'seed': seed,
                           'factored.refit_from': join(init_dir,
                                                       '%s_%i_%.0e.pkl' %
                                                       (decomposition, seed,
                                                        alpha)),
                           'factored.refit_data': [],
                           'full': True}
                          for seed in seeds
                          for alpha in [1e-3, 1e-4, 1e-5]
                          for decomposition in ['dl_random']]

    elif grid == 'factored_single':
        exp.config(factored_single)
        config_updates = ParameterGrid({'data.studies': studies,
                                        'seed': seeds})

    elif grid == 'factored_transfer':
        exp.config(factored_transfer)
        config_updates = ParameterGrid({'model.target_study': studies,
                                        'seed': seeds})
    elif grid == 'factored_l2_no_rank':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.seed': [100],
                                        'factored.latent_size': ['auto'],
                                        'factored.dropout': [0],
                                        'factored.init': ['normal'],
                                        'factored.adaptive_dropout': [False],
                                        'factored.l2_penalty': np.logspace(-4,
                                                                           -1,
                                                                           4)
                                        })

    elif grid == 'weight_power':
        exp.config(factored)
        weight_power = np.linspace(0, 1, 11)
        config_updates = ParameterGrid({'factored.weight_power': weight_power,
                                        'seed': seeds})

    elif grid == 'dropout':
        exp.config(factored)
        dropout = [0.5, 0.75]
        adaptive_dropout = [False, True]
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.dropout': dropout,
                                        'factored.adaptive_dropout':
                                            adaptive_dropout})
    elif grid == 'bn':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.batch_norm': [False]})
    elif grid == 'adaptive_dropout':
        exp.config(factored)
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored.adaptive_dropout': [False]})
    elif grid == 'trace':
        exp.config(trace)
        config_updates = ParameterGrid({'seed': seeds,
                                        'trace.trace_penalty':
                                            np.logspace(-4, 0, 5)})
    elif grid == 'trace_baseline':
        exp.config(trace)
        config_updates = ParameterGrid(
            {'seed': seeds, 'model.estimator': ['factored']})
    elif grid in ['logistic_gm', 'full_logistic']:
        if grid == 'logistic_gm':
            exp.config(logistic)
        elif grid == 'full_logistic':
            exp.config(full_logistic)
        config_updates = [{'seed': seed,
                           'data.studies': study} for seed in seeds
                          for study in studies]
    elif grid in ['logistic_tc', 'full_logistic_tc']:
        if grid == 'logistic_tc':
            exp.config(logistic)
        elif grid == 'full_logistic_tc':
            exp.config(full_logistic)
        exp.config(training_curve)
        config_updates = [{'seed': seed,
                           'data.train_size': {study: size},
                           'data.test_size': {study: .5},
                           'data.studies': study}
                          for seed in seeds
                          for study in ['brainomics', 'archi', 'henson2010faces', 'camcan']
                          for size in np.linspace(0.1, 0.5, 5)]
    elif grid in ['logistic_gm_full', 'full_logistic_full']:
        if grid == 'logistic_gm_full':
            exp.config(logistic)
        elif grid == 'full_logistic_full':
            exp.config(full_logistic)
        config_updates = ParameterGrid({'seed': [0],
                                        'data.studies': studies,
                                        'full': [True]
                                        })
    else:
        raise ValueError('Wrong argument')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # else:
    #     raise ValueError('Directory exists.')

    _id = get_id(output_dir)
    Parallel(n_jobs=50, verbose=100, backend='multiprocessing')(
        delayed(run_exp)(output_dir,
                         config_update,
                         mock=False,
                         _id=_id + i)
        for i, config_update
        in enumerate(list(config_updates)))
