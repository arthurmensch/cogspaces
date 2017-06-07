import os
from math import sqrt
from os.path import join

import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam, RMSprop

import keras.backend as K
from modl.utils.math.enet import enet_projection
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import load, dump
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.utils import gen_batches, check_random_state

from cogspaces.model import make_model, init_tensorflow, make_adversaries
from cogspaces.model_selection import StratifiedGroupShuffleSplit
from cogspaces.utils import get_output_dir

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
                # if batch_size is None:
                #     batches_generator[dataset] = repeat(slice(None))
                #     batch = slice(0, len_dataset)
                # else:
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
    datasets = ['archi', 'hcp']
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
    alpha = 1e-4
    latent_dim = 200
    activation = 'linear'
    source = 'hcp_rs_positive'
    optimizer = 'rmsprop'
    lr = 1e-3
    dropout_input = 0.25
    dropout_latent = 0.8
    batch_size = 128
    per_dataset_std = False
    joint_training = True
    epochs = 1000
    depth_weight = [0., 1., 0.]
    residual = False
    l1_proj = True
    patience = 10
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
                patience,
                dropout_latent,
                optimizer,
                l1_proj,
                activation,
                datasets,
                dataset_weight,
                residual,
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
            if dataset in ['brainomics']:
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
    y = np.argmax(y_oh, axis=1)
    y_oh = pd.DataFrame(index=X.index, data=y_oh)
    y = pd.DataFrame(index=X.index, data=y)
    dump(lbin, join(artifact_dir, 'lbin.pkl'))

    x_test = X.iloc[test]
    y_test = y.iloc[test]
    y_oh_test = y_oh.iloc[test]

    sample_weight_test = []
    for dataset, this_x in x_test.groupby(level='dataset'):
        sample_weight_test.append(pd.Series(np.ones(this_x.shape[0])
                                            / this_x.shape[0]
                                            * dataset_weight[dataset],
                                            index=this_x.index))
    sample_weight_test = pd.concat(sample_weight_test, axis=0)
    sample_weight_test /= np.min(sample_weight_test)

    x_test = x_test.values
    y_test = y_test.values
    y_oh_test = y_oh_test.values
    sample_weight_test = sample_weight_test.values

    X_train = X.iloc[train]
    y_train = y.iloc[train]
    y_oh_train = y_oh.iloc[train]

    train_data = pd.concat([X_train, y_train, y_oh_train],
                           keys=['X', 'y', 'y_oh'],
                           names=['type'], axis=1)
    train_data.sort_index(inplace=True)

    if steps_per_epoch is None:
        steps_per_epoch = X_train.shape[0] // batch_size

    init_tensorflow(n_jobs=n_jobs, debug=False)

    adversaries = make_adversaries(label_pool)

    np.save(join(artifact_dir, 'adversaries'), adversaries)
    np.save(join(artifact_dir, 'classes'), lbin.classes_)

    if False:  # not geometric_reduction or latent_dim is None:
        model = LogisticRegression(solver='saga',
                                   penalty='l1',
                                   C=1 / (X_train.shape[0] * alpha),
                                   verbose=verbose,
                                   max_iter=epochs,
                                   n_jobs=n_jobs,
                                   multi_class='multinomial')
        y_oh_train_inverse = lbin.inverse_transform(y_oh_train.values)
        model.fit(X_train.values, y_oh_train_inverse)
        y_pred_oh = model.predict_proba(X.values)
        y_pred_oh = {depth: y_pred_oh for depth in [0, 1, 2]}
    else:
        model = make_model(X.shape[1],
                           alpha=alpha,
                           latent_dim=latent_dim,
                           activation=activation,
                           dropout_input=dropout_input,
                           dropout_latent=dropout_latent,
                           non_negative=non_negative,
                           residual=residual,
                           adversaries=adversaries,
                           seed=_seed,
                           shared_supervised=shared_supervised)
        if not shared_supervised:
            for i, this_depth_weight in enumerate(depth_weight):
                if this_depth_weight == 0:
                    model.get_layer(
                        'supervised_depth_%i' % i).trainable = False
        if optimizer == 'sgd':
            optimizer = SGD(lr=lr, momentum=0)
        elif optimizer == 'adam':
            optimizer = Adam(lr=lr)  # beta_2=0.9)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(lr=lr)
        model.compile(loss=['categorical_crossentropy'] * 3,
                      optimizer=optimizer,
                      loss_weights=depth_weight,
                      metrics=['accuracy'])
        callbacks = [TensorBoard(log_dir=join(artifact_dir, 'logs'),
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=True),
                     ]
        if joint_training:
            generator = train_generator(train_data,
                                        batch_size,
                                        dataset_weight=dataset_weight,
                                        mix=mix_batch,
                                        seed=_seed)
            epoch = 0
            n_batch = 0
            logs = {}
            # reduce_lr = ReduceLROnPlateau(patience=10, verbose=1,
            #                               epsilon=1e-4, min_lr=1e-7)
            # reduce_lr.set_model(model)
            # reduce_lr.on_train_begin()
            warmup = False
            for x_train, y_train, sample_weight_train in generator:
                K.set_value(model.optimizer.lr, (1000 * lr) / (n_batch + 1))
                train_loss = model.train_on_batch(x_train, y_train,
                                                  sample_weight_train)
                if l1_proj and not warmup:
                    weights = model.get_layer('latent').get_weights()[0]
                    temp = np.zeros(weights.shape[0], dtype=np.float32)
                    for k in range(weights.shape[1]):
                        enet_projection(weights[:, k], temp, radius=1,
                                        l1_ratio=1)
                        weights[:, k] = temp[:]
                        # print(enet_norm(weights[:, k], 1))
                    sparsity = np.mean(weights == 0)
                    model.get_layer('latent').set_weights([weights])
                n_batch += 1
                if n_batch % steps_per_epoch == 0:
                    epoch += 1
                    if epoch > epochs:
                        break
                    test_loss = model.test_on_batch([x_test, y_test],
                                                    [y_oh_test] * 3,
                                                    [sample_weight_test] * 3)
                    logs['val_loss'] = test_loss[0]
                    # reduce_lr.on_epoch_end(epoch, logs)
                    info = ['Epoch %i/%i' % (epoch, epochs)]
                    if l1_proj and not warmup:
                        info += ['sparsity %.3f' % sparsity]
                    info += ["train_%s:%.3f" % (name, loss)
                            for name, loss in zip(model.metrics_names,
                                                  train_loss)]

                    info += ["test_%s:%.3f" % (name, loss)
                             for name, loss in zip(model.metrics_names,
                                                   test_loss)]
                    print(' '.join(info))
                # if epoch == 50 and warmup:
                #     print('End warmup')
                #     weights = model.get_layer('latent').get_weights()[0]
                #     l1_norm = np.sum(np.abs(weights), axis=0)
                #     weights /= l1_norm[np.newaxis, :]
                #     model.get_layer('latent').set_weights([weights])
                #     for depth in range(3):
                #         weights, bias = model.get_layer('supervised_depth_%i'
                #                                         % depth).get_weights()
                #         weights *= l1_norm[:, np.newaxis]
                #         model.get_layer('supervised_depth_%i'
                #                                         % depth).set_weights([weights, bias])
                #     warmup = False


            if retrain:
                model.get_layer('latent').trainable = False
                model.get_layer('dropout_input').rate = 0
                model.get_layer('dropout').rate = 0
                model.compile(loss=['categorical_crossentropy'] * 3,
                              optimizer=optimizer,
                              loss_weights=depth_weight,
                              metrics=['accuracy'])
                model.fit_generator(train_generator(train_data, batch_size,
                                                    dataset_weight=dataset_weight,
                                                    mix=False,
                                                    seed=_seed),
                                    callbacks=callbacks,
                                    validation_data=([x_test, y_test],
                                                     [y_oh_test] * 3,
                                                     [sample_weight_test] * 3,
                                                     ) if validation else None,
                                    steps_per_epoch=steps_per_epoch,
                                    verbose=verbose,
                                    initial_epoch=epochs,
                                    epochs=epochs + 30)

        else:
            model.fit_generator(
                train_generator(train_data.loc[['hcp']], batch_size,
                                dataset_weight=dataset_weight,
                                mix=False,
                                seed=_seed),
                callbacks=callbacks,
                validation_data=([x_test, y_test],
                                 [y_oh_test] * 3,
                                 [sample_weight_test] * 3
                                 ) if validation else None,
                steps_per_epoch=steps_per_epoch,
                verbose=verbose,
                epochs=epochs - 10)
            model.get_layer('latent').trainable = False
            model.compile(loss=['categorical_crossentropy'] * 3,
                          optimizer=optimizer,
                          loss_weights=depth_weight,
                          metrics=['accuracy'])
            model.fit_generator(train_generator(train_data, batch_size,
                                                dataset_weight=dataset_weight,
                                                mix=False,
                                                seed=_seed),
                                callbacks=callbacks,
                                validation_data=([x_test, y_test],
                                                 [y_oh_test] * 3,
                                                 [sample_weight_test] * 3,
                                                 ) if validation else None,
                                steps_per_epoch=steps_per_epoch,
                                verbose=verbose,
                                initial_epoch=epochs - 10,
                                epochs=epochs)
        y_pred_oh = model.predict(x=[X.values, y.values])

    _run.info['score'] = {}
    depth_name = ['full', 'dataset', 'task']
    for depth in [0, 1, 2]:
        this_y_pred_oh = y_pred_oh[depth]
        this_y_pred_oh_df = pd.DataFrame(index=X.index,
                                         data=this_y_pred_oh)
        dump(this_y_pred_oh_df, join(artifact_dir,
                                     'y_pred_depth_%i.pkl' % depth))
        y_pred = lbin.inverse_transform(this_y_pred_oh)  # "0_0_0"
        prediction = pd.DataFrame({'true_label': y_tuple,
                                   'predicted_label': y_pred},
                                  index=X.index)
        prediction = pd.concat([prediction.iloc[train],
                                prediction.iloc[test]],
                               names=['fold'], keys=['train', 'test'])
        prediction.to_csv(join(artifact_dir,
                               'prediction_depth_%i.csv' % depth))
        res = validate(prediction)
        _run.info['score'][depth_name[depth]] = res
        print('Prediction at depth %s' % depth_name[depth], res)

    if True:  # geometric_reduction and latent_dim is not None:
        model.save(join(artifact_dir, 'model.keras'))
    else:
        dump(model, join(artifact_dir, 'model.pkl'))
