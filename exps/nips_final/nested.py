from os.path import join

import numpy as np
import pandas as pd
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import dump
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from cogspaces.model.non_convex_pytorch import TransferEstimator
from cogspaces.pipeline import get_output_dir, make_data_frame, split_folds, \
    MultiDatasetTransformer

idx = pd.IndexSlice

exp = Experiment('nested')
basedir = join(get_output_dir(), 'nested')
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    datasets = ['la5c', 'hcp']
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')
    source = 'hcp_rs_positive_single'
    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5, 'full': .5}
    train_size = dict(hcp=None, archi=30, la5c=50, brainomics=30,
                      camcan=100,
                      human_voice=None)
    dataset_weights = {'brainomics': 1, 'archi': 1, 'hcp': 1}
    model = 'factored'
    max_iter = 1
    verbose = 10
    seed = 20

    with_std = False
    with_mean = False
    per_dataset = True

    # Factored only
    n_components = 100

    batch_size = 128
    optimizer = 'adam'
    step_size = 1e-3

    alphas = np.logspace(-6, -1, 12)
    latent_dropout_rates = [0.2, 0.4, 0.6]
    input_dropout_rates = [0., 0.1, 0.2]
    dataset_weights_helpers = [[1]]

    n_splits = 1
    n_jobs = 1


@exp.capture
def fit_model(df_train, df_test, datasets, model,
              n_components,
              per_dataset,
              batch_size,
              n_splits,
              with_std, with_mean,
              optimizer, alphas, latent_dropout_rates, input_dropout_rates,
              dataset_weights_helpers,
              step_size, max_iter, n_jobs, _run):
    transformer = MultiDatasetTransformer(with_std=with_std,
                                          with_mean=with_mean,
                                          integer_coding=True,
                                          per_dataset=per_dataset)
    transformer.fit(df_train)
    Xs_train, ys_train = transformer.transform(df_train)

    Xs_test, ys_test = transformer.transform(df_test)
    X_train, Xs_train_helpers = Xs_train[0], Xs_train[1:]
    y_train, ys_train_helpers = ys_train[0], ys_train[1:]
    X_test = Xs_test[0]

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25)

    if model == 'logistic_l2':
        torch.set_num_threads(1)
        estimator = TransferEstimator(
            n_components=n_components,
            architecture='flat',
            batch_size=batch_size,
            optimizer=optimizer,
            max_iter=max_iter,
            step_size=step_size, n_jobs=1)
        estimator = GridSearchCV(estimator,
                                 cv=cv,
                                 n_jobs=n_jobs,
                                 param_grid={'alpha': alphas})
    elif model == 'logistic_dropout':
        torch.set_num_threads(1)
        estimator = TransferEstimator(
            n_components=n_components,
            architecture='flat',
            batch_size=batch_size,
            optimizer=optimizer,
            max_iter=max_iter,
            step_size=step_size, n_jobs=1)
        estimator = GridSearchCV(estimator,
                                 cv=cv,
                                 n_jobs=n_jobs,
                                 param_grid={'input_dropout_rate': input_dropout_rates})
    elif model == 'factored':
        torch.set_num_threads(1)
        estimator = TransferEstimator(
            alpha=0, n_components=n_components,
            architecture='factored',
            batch_size=batch_size,
            optimizer=optimizer,
            max_iter=max_iter,
            step_size=step_size,
            Xs_helpers=Xs_train_helpers,
            ys_helpers=ys_train_helpers, )
        estimator = GridSearchCV(estimator,
                                 n_jobs=n_jobs,
                                 param_grid={
                                     'latent_dropout_rate': latent_dropout_rates,
                                     'input_dropout_rate': input_dropout_rates,
                                     'dataset_weights_helpers':
                                         dataset_weights_helpers})
    elif model == 'logistic_l2_sklearn':
        n_samples = X_train.shape[0]
        estimator = LogisticRegressionCV(solver='saga', max_iter=max_iter,
                                         Cs=1. / alphas / n_samples,
                                         n_jobs=n_jobs,
                                         cv=cv,
                                         verbose=10)
    else:
        raise ValueError('Wrong model argument')

    estimator.fit(X_train, y_train)

    if model != 'sklearn':
        print(estimator.cv_results_)

    ys_pred_train = estimator.best_estimator_.predict_all(Xs_train)
    ys_pred_test = estimator.best_estimator_.predict_all(Xs_test)
    pred_df_train = transformer.inverse_transform(df_train, ys_pred_train)
    pred_df_test = transformer.inverse_transform(df_test, ys_pred_test)

    return pred_df_train, pred_df_test, estimator, transformer


@exp.automain
def main(datasets, source, reduced_dir, unmask_dir,
         test_size, train_size,
         _run, _seed):
    artifact_dir = join(_run.observers[0].basedir, str(_run._id))
    single = False
    if source in ['hcp_rs_positive_single']:
        source = 'hcp_rs_positive'
        single = True
    df = make_data_frame(datasets, source,
                         reduced_dir=reduced_dir,
                         unmask_dir=unmask_dir)
    if single:
        df = df.iloc[:, -512:]
    df_train, df_test = split_folds(df, test_size=test_size,
                                    train_size=train_size,
                                    random_state=_seed)
    pred_df_train, pred_df_test, estimator, transformer \
        = fit_model(df_train, df_test,)

    pred_contrasts = pd.concat([pred_df_test, pred_df_train],
                               keys=['test', 'train'],
                               names=['fold'], axis=0)
    true_contrasts = pred_contrasts.index.get_level_values('contrast').values
    res = pd.DataFrame({'pred_contrast': pred_contrasts,
                        'true_contrast': true_contrasts})
    res.to_csv(join(artifact_dir, 'prediction.csv'))
    match = res['pred_contrast'] == res['true_contrast']
    score = match.groupby(level=['fold', 'dataset']).aggregate('mean')
    score_mean = match.groupby(level=['fold']).aggregate('mean')

    score_dict = {}
    for fold, this_score in score_mean.iteritems():
        score_dict['%s_mean' % fold] = this_score
    for (fold, dataset), this_score in score.iteritems():
        score_dict['%s_%s' % (fold, dataset)] = this_score
    _run.info['score'] = score_dict

    try:
        dump(estimator, join(artifact_dir, 'estimator.pkl'))
    except TypeError:
        pass
    dump(transformer, join(artifact_dir, 'transformer.pkl'))
    print(score)
    print(score_mean)
