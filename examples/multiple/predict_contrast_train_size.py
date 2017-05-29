import sys
from os import path

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.optional import pymongo
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.utils import check_random_state, shuffle

sys.path.append(path.dirname(path.dirname
                             (path.dirname(path.abspath(__file__)))))

from examples.predict_contrast import predict_contrast_exp

predict_contrast_multi_exp = Experiment('predict_contrast_train_size',
                                        ingredients=[predict_contrast_exp])
collection = predict_contrast_multi_exp.path
observer = MongoObserver.create(db_name='amensch', collection=collection)
predict_contrast_multi_exp.observers.append(observer)


@predict_contrast_multi_exp.config
def config():
    n_jobs = 24
    n_seeds = 10
    seed = 2


def single_run(config_updates, _id, master_id):
    observer = MongoObserver.create(db_name='amensch', collection=collection)
    predict_contrast_exp.observers = [observer]

    @predict_contrast_exp.config
    def config():
        n_jobs = 1
        epochs = 200
        steps_per_epoch = 200
        dropout_input = 0.25
        dropout_latent = 0.5
        source = 'hcp_rs_concat'
        depth_prob = [0, 1., 0]
        shared_supervised = False
        batch_size = 128
        alpha = 1e-5
        validation = False
        mix_batch = False
        verbose = 0
        train_size = dict(hcp=None, archi=None, la5c=None, brainomics=None,
                          camcan=None,
                          human_voice=None)

    run = predict_contrast_exp._create_run(
        config_updates=config_updates)
    run._id = _id
    run.info['master_id'] = master_id
    try:
        run()
    except:
        pass


@predict_contrast_multi_exp.automain
def run(n_seeds, n_jobs, _run, _seed):
    seed_list = check_random_state(_seed).randint(np.iinfo(np.uint32).max,
                                                  size=n_seeds)
    exps = []
    for dataset in ['archi', 'brainomics']:
        train_sizes = [dict(hcp=None, archi=i, la5c=None, brainomics=i,
                            camcan=i,
                            human_voice=None) for i in [5, 10, 20, 30, None]]
        transfer = [{'datasets': [dataset, 'hcp'],
                     'geometric_reduction': True,
                     'latent_dim': 50,
                     'dropout_input': 0.25,
                     'dropout_latent': 0.5,
                     'train_size': train_size,
                     'optimizer': 'adam',
                     'seed': seed} for seed in seed_list
                    for train_size in train_sizes]
        simple = [{'datasets': [dataset],
                   'geometric_reduction': True,
                   'latent_dim': 50,
                   'dropout_input': 0.25,
                   'dropout_latent': 0.5,
                   'train_size': train_size,
                   'optimizer': 'adam',
                   'seed': seed} for seed in seed_list
                  for train_size in train_sizes]
        exps += transfer
        exps += simple
    train_sizes = [dict(hcp=None, archi=None, la5c=None, brainomics=None,
                        camcan=i,
                        human_voice=None) for i in [5, 10, 20, 30, 100, 200,
                                                    None]]
    datasets_list = [['archi', 'camcan', 'brainomics'],
                     ['archi', 'hcp', 'camcan', 'brainomics'],
                     ['camcan'],
                     ['camcan', 'hcp']
                    ]
    transfer_camcan = [{'datasets': dataset,
                        'geometric_reduction': True,
                        'latent_dim': 50,
                        'dropout_input': 0.25,
                        'dropout_latent': 0.5,
                        'train_size': train_size,
                        'optimizer': 'adam',
                        'seed': seed} for seed in seed_list
                       for train_size in train_sizes
                       for dataset in datasets_list]
    exps += transfer_camcan

    # Robust labelling of experiments
    client = pymongo.MongoClient()
    database = client['amensch']
    c = database[collection].find({}, {'_id': 1})
    c = c.sort('_id', pymongo.DESCENDING).limit(1)
    c = c.next()['_id'] + 1 if c.count() else 1
    exps = shuffle(exps)

    Parallel(n_jobs=n_jobs,
             verbose=10)(delayed(single_run)(config_updates, c + i, _run._id)
                         for i, config_updates in enumerate(exps))
