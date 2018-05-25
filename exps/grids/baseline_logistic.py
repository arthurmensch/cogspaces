import numpy as np
import pandas as pd
from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id, OurFileStorageObserver
from exps.train import exp
from joblib import Parallel, delayed
from os.path import join
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state


@exp.config
def baseline():
    seed = 1
    system = dict(
        device=-1,
        verbose=100,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'masked'),
        studies='archi'
    )
    model = dict(
        normalize=True,
        estimator='logistic',
        max_iter=10000,
    )
    logistic = dict(
        l2_penalty=1e-6,
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


def main():
    source_dir = join(get_data_dir(), 'reduced_512')
    data, target = load_data_from_dir(data_dir=source_dir)
    studies = list(data.keys())
    l2_penalties = np.logspace(-4, 0, 20)

    seeds = check_random_state(1).randint(0, 100000, size=20)

    config_updates = ParameterGrid({'logistic.l2_penalty': l2_penalties,
                                    'data.studies': studies,
                                    'seed': seeds})
    output_dir = join(get_output_dir(), 'baseline_logistic')

    _id = get_id(output_dir)

    Parallel(n_jobs=20, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))

def refit():
    source_dir = join(get_data_dir(), 'reduced_512')
    data, target = load_data_from_dir(data_dir=source_dir)

    baseline_res = pd.read_pickle(join(get_output_dir(), 'baseline_avg.pkl'))


    studies = baseline_res.index.get_level_values('study').tolist()
    l2_penalties = baseline_res[('l2_penalty_', 'mean')].values.tolist()
    config_updates = [{'data.studies': study, 'logistic.l2_penalty': l2_penalty,
                       'full': True
                      } for study, l2_penalty in zip(studies, l2_penalties)]

    output_dir = join(get_output_dir(), 'baseline_logistic_refit')

    _id = get_id(output_dir)

    Parallel(n_jobs=20, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))

def full():
    source_dir = join(get_data_dir(), 'reduced_512')
    data, target = load_data_from_dir(data_dir=source_dir)
    studies = list(data.keys())
    l2_penalties = np.logspace(-4, 0, 20)

    seeds = check_random_state(1).randint(0, 100000, size=3)

    config_updates = ParameterGrid({'logistic.l2_penalty': l2_penalties,
                                    'data.studies': studies,
                                    'seed': seeds})
    output_dir = join(get_output_dir(), 'baseline_logistic_full')

    _id = get_id(output_dir)

    Parallel(n_jobs=10, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))


if __name__ == '__main__':
    full()