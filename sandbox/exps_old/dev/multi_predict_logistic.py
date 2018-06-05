import sys

import numpy as np
import os
from cogspaces.pipeline import get_output_dir
from os import path
from os.path import join
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_random_state

# Add examples to known models
sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))
from exps_old.old.exp_predict import exp as single_exp

exp = Experiment('predict_multi')
basedir = join(get_output_dir(), 'predict_multi')
if not os.path.exists(basedir):
    os.makedirs(basedir)
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    n_jobs = 1
    n_seeds = 1
    seed = 2


@single_exp.config
def config():
    datasets = ['archi', 'brainomics']
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')
    source = 'hcp_rs_positive_single'
    n_subjects = None
    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5}
    train_size = {'hcp': .9, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                  'la5c': .5}
    alpha = 0
    model = 'logistic'
    max_iter = 600
    n_components = 50
    latent_dropout_rate = 0.
    input_dropout_rate = 0.25
    source_init = None
    optimizer = 'adam'
    step_size = 1e-3

    verbose = 10
    with_std = True
    with_mean = True
    row_standardize = False


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
    exps = [{'datasets': ['archi'],
                 'beta': beta,
                 'seed': seed} for seed in seed_list
                for beta in np.logspace(-10, 0, 11)
                ]
    rundir = join(basedir, str(_run._id), 'run')
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, rundir, i)
                         for i, config_updates in enumerate(exps))
