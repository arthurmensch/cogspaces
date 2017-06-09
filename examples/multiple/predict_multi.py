import sys
import os
from os import path
from os.path import join

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_random_state

from cogspaces.utils import get_output_dir

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.predict_trace_norm import exp as single_exp

exp = Experiment('predict_contrast_trace_multi', ingredients=[single_exp])

basedir = join(get_output_dir(), 'multi_3')

if not os.path.exists(basedir):
    os.makedirs(basedir)

@exp.config
def config():
    n_jobs = 36
    n_seeds = 10
    seed = 2


def single_run(config_updates, _id):
    observer = FileStorageObserver.create(basedir=basedir)
    single_exp.observers = [observer]

    @single_exp.config
    def config():
        datasets = ['archi', 'hcp']
        reduced_dir = join(get_output_dir(), 'reduced')
        unmask_dir = join(get_output_dir(), 'unmasked')
        source = 'hcp_rs_positive'
        n_subjects = None
        test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                     'la5c': .5}
        train_size = {'hcp': .9, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                      'la5c': .5}
        alpha = 1e-3
        beta = 0
        max_iter = 3000
        verbose = 0

    run = single_exp._create_run(config_updates=config_updates)
    run._id = _id
    try:
        run()
    except:
        pass


@exp.automain
def run(n_seeds, n_jobs, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    exps = []
    for dataset in ['archi']:
        no_transfer = [{'datasets': [dataset],
                        'alpha': alpha,
                        'beta': beta,
                        'seed': seed} for seed in seed_list
                       for alpha in [0] + np.logspace(-5, 0, 5).tolist()
                       for beta in [0, 1e-4]]
        transfer = [{'datasets': [dataset, 'hcp'],
                     'alpha': alpha,
                     'beta': beta,
                     'seed': seed} for seed in seed_list
                    for alpha in [0] + np.logspace(-5, 0, 5).tolist()
                    for beta in [0, 1e-4]]
        exps += no_transfer
        exps += transfer

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, i)
                         for i, config_updates in enumerate(exps))
