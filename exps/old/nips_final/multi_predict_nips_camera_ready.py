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
# Add examples to known models
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from exps.old.exp_predict import exp as single_exp

exp = Experiment('nips')
basedir = join(get_output_dir(), 'nips')
if not os.path.exists(basedir):
    os.makedirs(basedir)
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    n_jobs = 10
    n_seeds = 10
    seed = 100


@single_exp.config
def config():
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')

    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5}
    train_size = dict(hcp=None, archi=30, la5c=50, brainomics=30,
                      camcan=100,
                      human_voice=None)

    max_iter = 1000
    verbose = 10
    seed = 10

    with_std = False
    with_mean = False
    per_dataset = False
    split_loss = True

    # Factored only
    n_components = 75
    alpha = 0.
    latent_dropout_rate = 0.5
    input_dropout_rate = 0.25
    batch_size = 128
    optimizer = 'adam'
    step_size = 1e-3


def single_run(config_updates, rundir, _id):
    run = single_exp._create_run(config_updates=config_updates)
    observer = FileStorageObserver.create(basedir=rundir)
    run._id = _id
    run.observers = [observer]
    try:
        run()
    except:
        pass


@exp.automain
def run(n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    exps = []
    for source in ['hcp_rs_positive', 'hcp_rs_positive_single']:
        for dataset in ['archi', 'brainomics', 'camcan', 'la5c']:
            # Multinomial model
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
            # Latent space model
            no_transfer = [{'datasets': [dataset],
                            'source': source,
                            'model': 'factored',
                            'with_std': True,
                            'with_mean': True,
                            'per_dataset': True,
                            'seed': seed} for seed in seed_list
                           ]
            transfer = [{'datasets': [dataset, 'hcp'],
                         'source': source,
                         'model': 'factored',
                         'with_std': True,
                         'with_mean': True,
                         'per_dataset': True,
                         'seed': seed} for seed in seed_list
                        ]
            # exps += no_transfer
            exps += transfer
            # exps += multinomial_dropout
            # exps += multinomial

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

    rundir = join(basedir, str(_run._id), 'run')
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, rundir, i)
                         for i, config_updates in enumerate(exps))
