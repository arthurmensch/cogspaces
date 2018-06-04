import sys
import time

import os
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id, OurFileStorageObserver
from exps.train import exp
from joblib import Parallel, delayed
from os.path import join
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state


def seed_split_init():
    seed = 10
    full = False
    system = dict(
        device=-1,
        verbose=0,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='all',
        target_study='archi'
    )
    model = dict(
        normalize=False,
        estimator='factored_variational',
        study_weight='sqrt_sample',
        max_iter={'pretrain': 300, 'sparsify': 0, 'finetune': 200},
    )
    factored_variational = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        epoch_counting='all',
        sampling='random',
        weight_power=0.6,
        batch_size=128,
        dropout=0.5,
        lr=1e-3,
        input_dropout=0.25)


def run_exp(output_dir, config_updates, _id, sleep, mock=False):
    """Boiler plate function that has to be put in every multiple
        experiment script, as exp does not pickle."""
    if not mock:
        observer = OurFileStorageObserver.create(basedir=output_dir)

        run = exp._create_run(config_updates=config_updates, )
        run._id = _id
        run.observers.append(observer)
        time.sleep(sleep)
        run()
    else:
        exp.run_command('print_config', config_updates=config_updates, )


if __name__ == '__main__':
    grid = sys.argv[1]
    if grid == 'seed_split_init':
        output_dir = join(get_output_dir(), 'seed_split_init')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(seed_split_init)
        model_seeds = check_random_state(42).randint(0, 100000, size=100)
        seeds = check_random_state(43).randint(0, 100000, size=20)
        config_updates = ParameterGrid({'model.seed': model_seeds,
                                        'full': [False]})
    else:
        raise ValueError('Wrong argument')

    _id = get_id(output_dir)
    Parallel(n_jobs=40, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      mock=False,
                                                      sleep=i,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))
