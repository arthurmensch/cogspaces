import os
import sys
from os import path
from os.path import join

import numpy as np
from cogspaces.pipeline import get_output_dir
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_random_state

# Add examples to known modules
sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))
from examples.predict import exp as single_exp

exp = Experiment('predict_log_multi')
basedir = join(get_output_dir(), 'predict_logistic_multi')
if not os.path.exists(basedir):
    os.makedirs(basedir)
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    n_jobs = 36
    n_seeds = 10
    seed = 2


@single_exp.config
def config():
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')
    n_subjects = None
    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5}
    train_size = {'hcp': .9, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                  'la5c': .5}
    model = 'logistic'
    max_iter = 1000
    verbose = 50


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
    for dataset in ['archi']:
        exps += [{'datasets': [dataset],
                  'beta': beta,
                  'source': source,
                  'seed': seed} for seed in seed_list
                 for beta in [0] + np.logspace(-5, -1, 4).tolist()
                 for source in ['hcp_rs_positive_single']]

    rundir = join(basedir, str(_run._id), 'run')
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, rundir, i)
                         for i, config_updates in enumerate(exps))
