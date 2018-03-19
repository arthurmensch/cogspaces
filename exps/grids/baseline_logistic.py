import cloudpickle

from os.path import join

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.utils.sacred import get_id, OurFileStorageObserver

from exps.train import exp


@exp.config
def baseline():
    seed = 0
    system = dict(
        device=-1,
        verbose=100,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_icbm_gm'),
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


if __name__ == '__main__':
    source_dir = join(get_data_dir(), 'reduced_512')
    data, target = load_data_from_dir(data_dir=source_dir)
    studies = list(data.keys())
    l2_penalties = np.logspace(-4, -1, 20)

    config_updates = ParameterGrid({'logistic.l2_penalty': l2_penalties,
                                    'data.studies': studies})
    output_dir = join(get_output_dir(), 'baseline_logistic_icbm_gm_2')

    _id = get_id(output_dir)

    Parallel(n_jobs=40, verbose=100)(delayed(run_exp)(output_dir,
                                                      config_update,
                                                      _id=_id + i)
                                     for i, config_update
                                     in enumerate(config_updates))
