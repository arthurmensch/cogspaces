import gc
import os

from os.path import join, expanduser

import numpy as np
import sys
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id, OurFileStorageObserver
from exps.train import exp
import pandas as pd


@exp.config
def base():
    seed = 0
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_lstsq'),
        studies=['hcp']
    )


def trace():
    model = dict(
        normalize=True,
        estimator='trace',
        study_weight='sqrt',
        max_iter=300,
    )
    trace = dict(
        trace_penalty=1e-2,
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
        optimizer='adam',
        shared_embedding_size=100,
        private_embedding_size=0,
        shared_embedding='hard+adversarial',
        skip_connection=False,
        batch_size=128,
        dropout=0.75,
        lr=1e-3,
        input_dropout=0.5,
    )


def factored():
    system = dict(
        device=-1,
        seed=0,
        verbose=50,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_icbm_gm'),
        studies='all'
    )

    model = dict(
        normalize=True,
        estimator='factored',
        study_weight='study',
        max_iter=50,
    )

    factored = dict(
        optimizer='sgd',
        shared_embedding_size='auto',
        private_embedding_size=0,
        shared_embedding='hard',
        skip_connection=False,
        batch_size=128,
        dropout=0.75,
        lr=1e-2,
        input_dropout=0.25,
    )


def all_pairs_4():
    seed = 1
    system = dict(
        device=-1,
        verbose=2,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_lstsq'),
        studies=['hcp']
    )
    model = dict(
        normalize=True,
        estimator='factored',
        study_weight='study',
        max_iter=500,
    )
    factored = dict(
        optimizer='adam',
        shared_embedding_size=100,
        private_embedding_size=0,
        shared_embedding='hard',
        skip_connection=False,
        activation='linear',
        cycle=True,
        batch_size=128,
        dropout=0.75,
        lr=1e-3,
        input_dropout=0.25,
    )


def all_pairs_advers():
    seed = 1
    system = dict(
        device=-1,
        verbose=2,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_lstsq'),
        studies=['hcp']
    )
    model = dict(
        normalize=True,
        estimator='factored',
        study_weight='study',
        max_iter=500,
    )
    factored = dict(
        optimizer='adam',
        shared_embedding_size=100,
        private_embedding_size=0,
        shared_embedding='adversarial',
        skip_connection=False,
        activation='linear',
        cycle=True,
        batch_size=128,
        dropout=0.75,
        lr=1e-3,
        input_dropout=0.25,
    )


def positive_transfer():
    data = dict(study_weight='target')


def run_exp(output_dir, config_updates, _id, mock=False):
    """Boiler plate function that has to be put in every multiple
        experiment script, as exp does not pickle."""
    if not mock:
        exp.run_command('print_config', config_updates=config_updates, )
        observer = OurFileStorageObserver.create(basedir=output_dir)

        run = exp._create_run(config_updates=config_updates, )
        run._id = _id
        run.observers.append(observer)
        run()
    else:
        exp.run_command('print_config', config_updates=config_updates, )
    run = None
    gc.collect()


def get_studies_list(exp='all_pairs_4'):
    res = pd.read_pickle(join(expanduser('~/output/cogspaces/%s.pkl' % exp)))

    baseline = res.reset_index()
    baseline = baseline.loc[baseline['target'] == baseline['help']]
    baseline.set_index(['target', 'seed'], inplace=True)
    baseline.drop(columns=['help'], inplace=True)
    baseline.sort_index(inplace=True)

    result = pd.merge(res.reset_index(), baseline.reset_index(),
                      suffixes=('', '_baseline'),
                      on=['target', 'seed'], how='outer').set_index(
        ['target', 'help', 'seed'])
    result.sort_index(inplace=True)
    result['diff'] = result['score'] - result['score_baseline']
    diff = result['diff'].groupby(level=['target', 'help']).agg(
        {'mean': np.mean, 'std': np.std})
    studies_list = diff.index.get_level_values('target').unique().values

    def trim_neg_and_norm(x):
        x = x.loc[x > 0].reset_index('target', drop=True)
        return x

    positive = diff['mean'].groupby('target').apply(trim_neg_and_norm)

    res = []
    for target in studies_list:
        try:
            help = positive.loc[target]
        except:
            help = []
        studies = [target] + help.index.get_level_values(
            'help').values.tolist()
        res.append(studies)
    return res


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
    elif grid == 'factored_5':
        output_dir = join(get_output_dir(), 'factored_5')
        exp.config(factored)
        config_updates = []
        for optimizer in ['adam', 'sgd']:
            config_updates += list(
                ParameterGrid({'factored.dropout': [0.75, 0.875],
                               'factored.shared_embedding_size': [256, 512],
                               'factored.private_embedding_size': [0],
                               'factored.shared_embedding': ['hard'],
                               'factored.optimizer': [optimizer],
                               'factored.lr': [1e-3, 2e-3, 5e-3]
                               if optimizer == 'sgd' else [1e-3],
                               'model.study_weight': ['sqrt_sample']
                               }))
    elif grid == 'factored_4':
        output_dir = join(get_output_dir(), 'factored_4')
        exp.config(factored)
        config_updates = []
        config_updates += list(
            ParameterGrid({'factored.dropout': [15 / 16],
                           'factored.shared_embedding_size': [512],
                           'factored.private_embedding_size': [0],
                           'factored.shared_embedding':
                               ['hard'],
                           'factored.optimizer': ['adam', 'sgd'],
                           'factored.lr': [5e-3, 1e-3],
                           'model.study_weight':
                               ['sqrt_sample']
                           }))
    elif grid == 'all_pairs_4':
        output_dir = join(get_output_dir(), 'all_pairs_4')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(all_pairs_4)
        source_dir = join(get_data_dir(), 'reduced_512_lstsq')
        data, target = load_data_from_dir(data_dir=source_dir)
        studies_list = list(data.keys())
        n_studies = len(studies_list)
        config_updates = []
        seeds = check_random_state(1).randint(0, 100000, size=20)
        for seed in seeds:
            for i in range(n_studies):
                for j in range(i):
                    studies = [studies_list[i], studies_list[j]]
                    config_updates.append({'data.studies': studies,
                                           'seed': seed})
                config_updates.append({'data.studies': [studies_list[i]],
                                       'seed': seed})
    elif grid == 'positive_transfer':
        output_dir = join(get_output_dir(), 'positive_transfer')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        exp.config(all_pairs_4)
        exp.config(positive_transfer)
        source_dir = join(get_data_dir(), 'reduced_512_lstsq')
        studies_list = get_studies_list()
        config_updates = []
        seeds = check_random_state(1).randint(0, 100000, size=20)
        for seed in seeds:
            for studies in studies_list:
                config_updates.append({'data.studies': studies,
                                       'seed': seed})
                config_updates.append({'data.studies': [studies[0]],
                                       'seed': seed})
    elif grid == 'all_pairs_advers':
        output_dir = join(get_output_dir(), 'all_pairs_advers')
        exp.config(all_pairs_advers)
        source_dir = join(get_data_dir(), 'reduced_512_lstsq')
        data, target = load_data_from_dir(data_dir=source_dir)
        studies_list = list(data.keys())
        n_studies = len(studies_list)
        config_updates = []
        seeds = check_random_state(1).randint(0, 100000, size=20)
        for seed in seeds:
            for i in range(n_studies):
                for j in range(i):
                    studies = [studies_list[i], studies_list[j]]
                    config_updates.append({'data.studies': studies,
                                           'seed': seed})
                config_updates.append({'data.studies': [studies_list[i]],
                                       'seed': seed})

    else:
        raise ValueError('Wrong argument')
    _id = get_id(output_dir)
    Parallel(n_jobs=30, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      mock=False,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))
