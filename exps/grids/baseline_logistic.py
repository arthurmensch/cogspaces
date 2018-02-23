from os.path import join

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import OurFileStorageObserver
from exps.train import exp


def baseline():
    system = dict(
        device=-1,
        seed=0,
        verbose=100,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='archi'
    )
    model = dict(
        normalize=True,
        estimator='logistic',
        max_iter=10,
    )
    logistic = dict(
        l2_penalty=1e-6,
    )


def run_exp(exp, config_updates, _id):
    output_dir = join(get_output_dir(), 'baseline_logistic')
    exp.run_command('print_config', config_updates=config_updates,
                    named_configs=['baseline'])
    run = exp._create_run(config_updates=config_updates,
                          named_configs=['baseline'])
    observer = OurFileStorageObserver.create(basedir=output_dir)
    run.observers.append(observer)
    run()


if __name__ == '__main__':
    exp.named_config(baseline)
    source_dir = join(get_data_dir(), 'reduced_512')
    data, target = load_data_from_dir(data_dir=source_dir)
    studies = list(data.keys())[:2]
    l2_penalties = np.logspace(-4, -1, 20)[:2]

    config_updates = ParameterGrid({'logistic.l2_penalty': l2_penalties,
                                    'data.studies': studies})
    Parallel(n_jobs=1, verbose=100)(delayed(run_exp)(exp, config_update,
                                                     _id=_id)
                                    for _id, config_update
                                    in enumerate(config_updates))
