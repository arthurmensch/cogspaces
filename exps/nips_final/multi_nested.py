import os
import sys
from os import path
from os.path import join

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_random_state

from cogspaces.pipeline import get_output_dir

print(path.dirname(path.dirname(path.abspath(__file__))))
# Add examples to known modules
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from exps.nips_final.nested import exp as single_exp

exp = Experiment('multi_nested')
basedir = join(get_output_dir(), 'multi_nested')
if not os.path.exists(basedir):
    os.makedirs(basedir)
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    n_jobs = 20
    n_seeds = 10
    seed = 1


@single_exp.config
def config():
    datasets = ['archi', 'hcp']
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')
    source = 'hcp_rs_positive_single'
    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5, 'full': .5}
    train_size = dict(hcp=None, archi=30, la5c=50, brainomics=30,
                      camcan=100,
                      human_voice=None)
    dataset_weights = {'brainomics': 1, 'archi': 1, 'hcp': 1}
    max_iter = 1000
    verbose = 10
    seed = 20

    with_std = True
    with_mean = True
    per_dataset = True

    # Factored only
    n_components = 100

    batch_size = 128
    optimizer = 'adam'
    step_size = 1e-3

    alphas = np.logspace(-6, -1, 9)
    latent_dropout_rates = [0.5]
    input_dropout_rates = [0.25]
    dataset_weights_helpers = [[1]]

    n_splits = 10
    n_jobs = 1


def single_run(config_updates, rundir, _id):
    run = single_exp._create_run(config_updates=config_updates)
    observer = FileStorageObserver.create(basedir=rundir)
    run._id = _id
    run.observers = [observer]
    run()


@exp.automain
def run(n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    exps = []
    for source in ['hcp_rs_concat', 'hcp_rs']:
        for dataset in ['archi', 'brainomics', 'camcan', 'la5c']:
            # Multinomial model
            multinomial_l2 = [{'datasets': [dataset],
                               'source': source,
                               'model': 'logistic_l2_sklearn',
                               'max_iter': 200,
                               # Multinomial l2 works better with no standardization
                               'with_std': False,
                               'with_mean': False,
                               'seed': seed} for seed in seed_list
                              ]
            multinomial_dropout = [{'datasets': [dataset],
                                       'source': source,
                                       'model': 'logistic_dropout',
                                       'seed': seed} for seed in seed_list
                                      ]
            # Latent space model
            no_transfer = [{'datasets': [dataset],
                            'source': source,
                            'model': 'factored',
                            'seed': seed} for seed in seed_list
                           ]
            transfer = [{'datasets': [dataset, 'hcp'],
                         'source': source,
                         'model': 'factored',
                         'seed': seed} for seed in seed_list
                        ]
            exps += no_transfer
            exps += transfer
            exps += multinomial_l2
            exps += multinomial_dropout

    # Slow (uncomment if needed)
    source = 'unmasked'

    multinomial = [{'datasets': [dataset],
                    'source': source,
                    'alpha': alpha,
                    'model': 'logistic',
                    'latent_dropout_rate': 0.,
                    'input_dropout_rate': 0.,
                    'seed': seed} for seed in seed_list
                   for alpha in np.logspace(-6, -1, 6)
                   ]
    multinomial_dropout = [{'datasets': [dataset],
                            'source': source,
                            'alpha': 0,
                            'model': 'logistic',
                            'latent_dropout_rate': 0.,
                            'input_dropout_rate': 0.25,
                            'seed': seed} for seed in seed_list
                           ]
    # exps += multinomial_dropout
    # exps += multinomial

    np.random.shuffle(exps)

    rundir = join(basedir, str(_run._id), 'run')
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, rundir, i)
                         for i, config_updates in enumerate(exps))
