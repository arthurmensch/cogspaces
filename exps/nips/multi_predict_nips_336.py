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
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from exps.exp_predict import exp as single_exp

exp = Experiment('nips_336')
basedir = join(get_output_dir(), 'nips_336')
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
    datasets = ['camcan', 'hcp']
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')
    source = 'hcp_rs_concat'
    n_subjects = None
    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5}
    train_size = dict(hcp=None, archi=30, la5c=50, brainomics=30,
                      camcan=100,
                      human_voice=None)
    alpha = 0
    beta = 0
    model = 'logistic'
    max_iter = 400
    n_components = 50
    latent_dropout_rate = 0.
    input_dropout_rate = 0.0
    source_init = None
    optimizer = 'adam'
    step_size = 1e-3

    verbose = 10
    with_std = True
    with_mean = True
    per_dataset = True


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
    for source in ['hcp_rs_concat']:
        for dataset in ['camcan', 'la5c', 'brainomics', 'archi']:
            no_transfer = [{'datasets': [dataset],
                            'source': source,
                            'beta': beta,
                            'seed': seed} for seed in seed_list
                           for beta in np.logspace(-6, -1, 6)
                           ]
            exps += no_transfer

    rundir = join(basedir, str(_run._id), 'run')
    if not os.path.exists(rundir):
        os.makedirs(rundir)

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, rundir, i)
                         for i, config_updates in enumerate(exps))
