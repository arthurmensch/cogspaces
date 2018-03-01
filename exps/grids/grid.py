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
        study_weight='sqrt',
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
        optimizer='sgd',
        shared_embedding_size=100,
        private_embedding_size=0,
        shared_embedding='hard+adversarial',
        skip_connection=False,
        batch_size=128,
        dropout=0.75,
        lr=1e-2,
        input_dropout=0.5,
    )


def factored_l2():
    model = dict(
        normalize=True,
        estimator='factored',
        max_iter=300,
    )

    factored = dict(
        optimizer='sgd',
        shared_embedding_size=100,
        private_embedding_size=0,
        shared_embedding='hard+adversarial',
        skip_connection=False,
        batch_size=128,
        dropout=0.75,
        lr=1e-2,
        input_dropout=0.5,
    )


def factored():
    system = dict(
        device=-1,
        seed=0,
        verbose=1000,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_icbm_gm'),
        studies='all'
    )

    model = dict(
        normalize=True,
        estimator='factored',
        study_weight='study',
        max_iter=1000,
    )

    factored = dict(
        optimizer='sgd',
        shared_embedding_size=128,
        private_embedding_size=0,
        shared_embedding='hard',
        skip_connection=False,
        batch_size=32,
        dropout=0.75,
        lr=1e-2,
        input_dropout=0.25,
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
    grid = sys.argv[1]
    if grid == 'trace':
        output_dir = join(get_output_dir(), 'trace')
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
        output_dir = join(get_output_dir(), 'factored_dropout')

        exp.config(factored_dropout)
        dropouts = [0.70, 0.80, 0.90]
        embedding_sizes = [100, 200, 300, 400, 'auto']
        study_weights = ['sqrt_sample', 'study']
        config_updates = ParameterGrid({'factored.dropout':
                                            dropouts,
                                        'factored.embedding_size':
                                            embedding_sizes,
                                        'model.study_weight': study_weights})
        _id = get_id(output_dir)
        Parallel(n_jobs=30, verbose=100)(delayed(run_exp)(output_dir,
                                                          config_update,
                                                          _id=_id + i)
                                         for i, config_update
                                         in enumerate(config_updates))
    elif grid == 'factored_l2':
        output_dir = join(get_output_dir(), 'factored_l2')
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
    elif grid == 'factored':
        output_dir = join(get_output_dir(), 'factored')
        exp.config(factored)
        config_updates = list(ParameterGrid({'factored.dropout': [0.75, 0.875],
                                             'factored.shared_embedding_size':
                                                 [128, 256],
                                             'factored.private_embedding_size':
                                                 [0, 16],
                                             'factored.shared_embedding':
                                                 ['hard', 'hard+adversarial'],
                                             'factored.optimizer':
                                                 ['adam', 'sgd'],
                                             'model.study_weight':
                                                 ['study', 'sqrt_sample']
                                             }))
        for config_update in config_updates:
            if config_update['factored.optimizer'] == 'adam':
                config_update['factored.lr'] = 1e-3
            else:
                config_update['factored.lr'] = 1e-2

        _id = get_id(output_dir)
        Parallel(n_jobs=1, verbose=100)(delayed(run_exp)(output_dir,
                                                          config_update,
                                                          _id=_id + i)
                                         for i, config_update
                                         in enumerate(config_updates))
    else:
        raise ValueError('Wrong argument')
