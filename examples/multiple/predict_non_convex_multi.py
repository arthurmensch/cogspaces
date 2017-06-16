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

exp = Experiment('predict_trace_multi')
basedir = join(get_output_dir(), 'predict_trace_multi')
if not os.path.exists(basedir):
    os.makedirs(basedir)
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    n_jobs = 24
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
    model = 'non_convex'
    max_iter = 300
    alpha = 0
    beta = 0
    verbose = 0


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
        exps += [{'datasets': [dataset, 'hcp'],
                  'n_components': n_components,
                  'latent_dropout_rate': latent_dropout_rate,
                  'source': source,
                  'with_std': with_std,
                  'seed': seed} for seed in seed_list
                 for latent_dropout_rate in [0, 0.25, 0.5, 0.75]
                 for n_components in [25, 50, 200]
                 for with_std in [True, False]
                 for source in ['hcp_rs_positive']]

    rundir = join(basedir, str(_run._id), 'run')
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, rundir, i)
                         for i, config_updates in enumerate(exps))
