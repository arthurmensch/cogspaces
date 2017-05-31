import os

from math import sqrt
from os.path import join

import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import load, dump
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.utils import gen_batches, check_random_state

from cogspaces.utils import get_output_dir
from cogspaces.model import make_model, init_tensorflow, make_adversaries
from cogspaces.model_selection import StratifiedGroupShuffleSplit

idx = pd.IndexSlice

predict_contrast_exp = Experiment('predict_contrast')

base_artifact_dir = join(get_output_dir(), 'predict')
observer = FileStorageObserver.create(basedir=base_artifact_dir)

predict_contrast_exp.observers.append(observer)


def scale(X, train, per_dataset_std):
    X_train = X.iloc[train]
    if per_dataset_std:
        standard_scaler = {}
        corr = np.sum(np.sqrt(
            X_train[0].groupby(level='dataset').aggregate('count').values))
        for dataset, this_X_train in X_train.groupby(level='dataset'):
            this_standard_scaler = StandardScaler()
            this_standard_scaler.fit(this_X_train)
            this_standard_scaler.scale_ /= sqrt(this_X_train.shape[0]) / corr
            standard_scaler[dataset] = this_standard_scaler
        new_X = []
        for dataset, this_X in X.groupby(level='dataset'):
            this_new_X = standard_scaler[dataset].transform(this_X)
            this_new_X = pd.DataFrame(this_new_X, this_X.index)
            new_X.append(this_new_X)
        X = pd.concat(new_X)
    else:
        standard_scaler = StandardScaler(with_std=False)
        standard_scaler.fit(X_train)
        X_new = standard_scaler.transform(X)
        X = pd.DataFrame(X_new, index=X.index)
    return X, standard_scaler


def train_generator(train_data, batch_size, dataset_weight,
                    mix, seed):
    random_state = check_random_state(seed)
    grouped_data = train_data.groupby(level='dataset')
    grouped_data = {dataset: sub_data for dataset, sub_data in
                    grouped_data}
    batches_generator = {}
    datasets = list(grouped_data.keys())
    datasets = [dataset for dataset in datasets for _ in
                range(dataset_weight[dataset])]
    n_dataset = len(grouped_data)
    x_batch = np.empty((batch_size, train_data['X'].shape[1]))
    y_batch = np.empty((batch_size, train_data['y'].shape[1]))
    y_oh_batch = np.empty((batch_size, train_data['y_oh'].shape[1]))
    sample_weight_batch = np.empty(batch_size)
    while True:
        start = 0
        for dataset in datasets:
            data_one_dataset = grouped_data[dataset]
            if not mix:
                start = 0
            len_dataset = data_one_dataset.shape[0]
            try:
                batch = next(batches_generator[dataset])
            except (KeyError, StopIteration):
                batches_generator[dataset] = gen_batches(len_dataset,
                                                         batch_size
                                                         // n_dataset if mix
                                                         else batch_size)
                permutation = random_state.permutation(len_dataset)
                data_one_dataset = data_one_dataset.iloc[permutation]
                grouped_data[dataset] = data_one_dataset
                batch = next(batches_generator[dataset])
            len_batch = batch.stop - batch.start
            stop = start + len_batch
            batch_data = data_one_dataset.iloc[batch]
            x_batch[start:stop] = batch_data['X'].values
            y_batch[start:stop] = batch_data['y'].values
            y_oh_batch[start:stop] = batch_data['y_oh'].values
            sample_weight_batch[start:stop] = np.ones(len_batch)
            start = stop
            if not mix:
                yield ([x_batch[:stop].copy(), y_batch[:stop].copy()],
                       [y_oh_batch[:stop].copy()] * 3,
                       [sample_weight_batch[:stop]] * 3)
        if mix:
            yield ([x_batch[:stop].copy(), y_batch[:stop].copy()],
                   [y_oh_batch[:stop].copy()] * 3,
                   [sample_weight_batch[:stop]] * 3)


@predict_contrast_exp.config
def config():
    datasets = ['archi', 'hcp', 'brainomics', 'camcan']
    test_size = dict(hcp=0.1, archi=0.5, la5c=0.5, brainomics=0.5,
                     camcan=.5,
                     human_voice=0.5)
    n_subjects = dict(hcp=None, archi=None, la5c=None, brainomics=None,
                      camcan=None,
                      human_voice=None)
    dataset_weight = dict(hcp=1, archi=1, la5c=1, brainomics=1,
                          camcan=1,
                          human_voice=1)
    train_size = dict(hcp=None, archi=None, la5c=None, brainomics=None,
                      camcan=None,
                      human_voice=None)
    validation = True
    geometric_reduction = True
    alpha = 1e-3
    latent_dim = 20
    activation = 'linear'
    source = 'hcp_rs_concat'
    optimizer = 'adam'
    lr = 1e-3
    dropout_input = 0.25
    dropout_latent = 0.5
    batch_size = 256
    per_dataset_std = False
    joint_training = True
    epochs = 50
    depth_weight = [0., 1., 0.]
    n_jobs = 2
    verbose = 2
    seed = 10
    shared_supervised = False
    retrain = False
    mix_batch = False
    non_negative = False
    steps_per_epoch = 100
    _seed = 0


def validate(prediction):
    match = prediction['true_label'] == prediction['predicted_label']
    prediction = prediction.assign(match=match)

    score = prediction['match'].groupby(level=['fold', 'dataset']).apply(
        np.mean)
    res = {}
    for fold, sub_score in score.groupby(level='fold'):
        res[fold] = {}
        for (_, dataset), this_score in sub_score.iteritems():
            res[fold][dataset] = this_score
    return res


@predict_contrast_exp.automain
def train_model(alpha,
                latent_dim,
                n_subjects,
                geometric_reduction,
                test_size,
                train_size,
                retrain,
                dropout_input,
                joint_training,
                lr,
                mix_batch,
                non_negative,
                source,
                dropout_latent,
                optimizer,
                activation,
                datasets,
                dataset_weight,
                steps_per_epoch,
                depth_weight,
                batch_size,
                epochs,
                verbose,
                shared_supervised,
                validation,
                n_jobs,
                _run,
                _seed):
    artifact_dir = join(base_artifact_dir, str(_run._id), 'artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    output_dir = get_output_dir()
    reduced_dir = join(output_dir, 'reduced')
    unmask_dir = join(output_dir, 'unmasked')
    if verbose:
        print('Fetch data')
    X = []
    keys = []

    for dataset in datasets:
        if dataset_weight[dataset] != 0:
            this_reduced_dir = join(reduced_dir, source, dataset)
            if geometric_reduction:
                this_X = load(join(this_reduced_dir, 'Xt.pkl'))
            else:
                this_X = load(join(unmask_dir, dataset, 'imgs.pkl'))
            if dataset in ['archi', 'brainomics']:
                this_X = this_X.drop(['effects_of_interest'],
                                     level='contrast', )
            subjects = this_X.index.get_level_values('subject'). \
                unique().values.tolist()
            subjects = subjects[:n_subjects[dataset]]
            this_X = this_X.loc[idx[subjects]]
            X.append(this_X)
            keys.append(dataset)

    X = pd.concat(X, keys=keys, names=['dataset'])

    X = X.reset_index(level=['direction'], drop=True)
    X.sort_index(inplace=True)

    # Cross validation folds
    cv = StratifiedGroupShuffleSplit(stratify_levels='dataset',
                                     group_name='subject',
                                     test_size=test_size,
                                     train_size=train_size,
                                     n_splits=1,
                                     random_state=0)
    train, test = next(cv.split(X))

    y = np.concatenate([X.index.get_level_values(level)[:, np.newaxis]
                        for level in ['dataset', 'task', 'contrast']],
                       axis=1)

    y_tuple = ['__'.join(row) for row in y]
    lbin = LabelBinarizer()
    y_oh = lbin.fit_transform(y_tuple)
    label_pool = lbin.classes_
    label_pool = [np.array(e.split('__')) for e in label_pool]
    label_pool = np.vstack(label_pool)
    y_oh = pd.DataFrame(index=X.index, data=y_oh)
    # y = np.argmax(y_oh, axis=1)
    # y = pd.DataFrame(index=X.index, data=y)
    dump(lbin, join(artifact_dir, 'lbin.pkl'))

    x_test = X.iloc[test]
    y_test = y.iloc[test]
    y_oh_test = y_oh.iloc[test]
