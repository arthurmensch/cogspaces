from os.path import join

import numpy as np
import sys
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id, OurFileStorageObserver
from exps.train import exp


@exp.config
def base():
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_lstsq'),
        studies='all'
    )


def trace():
    model = dict(
        normalize=True,
        estimator='trace',
        max_iter=300,
    )
    trace = dict(
        trace_penalty=5e-2,
    )


def factored_dropout():
    model = dict(
        normalize=True,
        estimator='factored',
        max_iter=300,
    )
    factored = dict(
        optimizer='adam',
        embedding_size=300,
        batch_size=128,
        dropout=0.0,
        input_dropout=0.25,
        l2_penalty=0,
        l1_penalty=0
    )


def factored_l2():
    model = dict(
        normalize=True,
        estimator='factored',
        max_iter=300,
    )

    factored = dict(
        optimizer='lbfgs',
        embedding_size='auto',
        batch_size=128,
        dropout=0.0,
        input_dropout=0.0,
        l2_penalty=0,
        l1_penalty=0
    )


def run_exp(output_dir, config_updates, _id):
    """Boiler plate function that has to be put in every multiple
        experiment script, as exp does not pickle."""
    exp.run_command('print_config', config_updates=config_updates, )
    run = exp._create_run(config_updates=config_updates, )
    run._id = _id
    observer = OurFileStorageObserver.create(basedir=output_dir)
    run.observers.append(observer)
    run()


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'multi_studies')
    grid = sys.argv[1]
    if grid == 'trace':
        exp.config(trace)
        trace_penalties = np.logspace(-4, -1, 15)
        config_updates = ParameterGrid({'trace.trace_penalty':
                                        trace_penalties})
        _id = get_id(output_dir)
        Parallel(n_jobs=15, verbose=100)(delayed(run_exp)(output_dir,
                                                          config_update,
                                                          _id=_id + i)
                                         for i, config_update
                                         in enumerate(config_updates))
    elif grid == 'factored_dropout':
        exp.config(factored_dropout)
        dropouts = [0.70, 0.75, 0.80, 0.85, 0.90]
        config_updates = ParameterGrid({'factored.dropout':
                                        dropouts})
        _id = get_id(output_dir)
        Parallel(n_jobs=5, verbose=100)(delayed(run_exp)(output_dir,
                                                         config_update,
                                                         _id=_id + i)
                                        for i, config_update
                                        in enumerate(config_updates))
    elif grid == 'factored_l2':
        exp.config(factored_l2)
        l2_penalties = np.logspace(-4, -1, 15)
        config_updates = ParameterGrid({'factored.l2_penalty':
                                        l2_penalties})
        _id = get_id(output_dir)
        Parallel(n_jobs=15, verbose=100)(delayed(run_exp)(output_dir,
                                                          config_update,
                                                          _id=_id + i)
                                         for i, config_update
                                         in enumerate(config_updates))
    else:
        raise ValueError('Wrong argument')