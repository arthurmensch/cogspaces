import sys
import time

import os
from joblib import Parallel, delayed
from os.path import join
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state

from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id, OurFileStorageObserver
from exps.train import exp


@exp.config
def base():
    seed = 0
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies=['hcp']
    )


def variational():
    seed = 1
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
        max_iter={'pretrain': 300, 'sparsify': 0, 'finetune': 0},
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
        input_dropout=0.1)




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
    if grid == 'big_gamble':
        output_dir = join(get_output_dir(), 'big_gamble')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(variational)
        seeds = check_random_state(1).randint(0, 100000, size=200)
        config_updates = ParameterGrid({'model.seed': seeds, 'full': [True]})
    elif grid == 'weight_power':
        output_dir = join(get_output_dir(), 'weight_power')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(variational)
        seeds = check_random_state(1).randint(0, 100000, size=5)
        weight_power = [0, 0.25, 0.5, 0.71, 1]
        config_updates = ParameterGrid({'seed': seeds,
                                        'factored_variational.weight_power': weight_power})
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
