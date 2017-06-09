import json
from os.path import join

import numpy as np
import pandas as pd
from cogspaces.model import TraceNormEstimator, NonConvexEstimator
from cogspaces.pipeline import get_output_dir, make_data_frame, split_folds, \
    MultiDatasetTransformer
from joblib import load
from numpy.linalg import svd
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import dump
from sklearn.linear_model import LogisticRegression

idx = pd.IndexSlice

exp = Experiment('Predict')
basedir = join(get_output_dir(), 'predict')
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    datasets = ['archi', 'hcp']
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')
    source = 'hcp_rs'
    n_subjects = None
    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5}
    train_size = {'hcp': .9, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                  'la5c': .5}
    model = 'logistic'
    alpha = 5e-4
    beta = 1e-5
    max_iter = 500
    verbose = 10
    seed = 10

    # Non convex only
    n_components = 200
    latent_dropout_rate = 0.5
    input_dropout_rate = 0.25
    source_init = join(get_output_dir(), 'clean', '557')
    optimizer = 'adam'
    step_size = 1e-2


@exp.capture
def fit_model(df_train, df_test, model, alpha, beta, n_components,
              optimizer, latent_dropout_rate, input_dropout_rate,
              step_size, source_init, max_iter, verbose):
    transformer = MultiDatasetTransformer()
    Xs_train, ys_train = transformer.fit_transform(df_train)
    Xs_test, ys_test = transformer.fit_transform(df_test)
    if model == 'logistic':  # Adaptation
        ys_pred_train = []
        ys_pred_test = []
        for X_train, X_test, y_train in zip(Xs_train, Xs_test, ys_train):
            _, n_targets = y_train.shape
            if beta == 0:
                beta = 1e-20
            estimator = LogisticRegression(C=1 / (X_train.shape[0] * beta),
                                           multi_class='multinomial',
                                           max_iter=max_iter,
                                           solver='saga',
                                           verbose=verbose)
            y_train = np.argmax(y_train, axis=1)
            estimator.fit(X_train, y_train)
            y_pred_train = estimator.predict(X_train)
            y_pred_test = estimator.predict(X_test)

            n_samples = X_train.shape[0]
            bin_y = np.zeros((y_pred_train.shape[0], n_targets), dtype='int64')
            for i in range(n_samples):
                bin_y[i, y_pred_train[i]] = 1
            y_pred_train = bin_y
            n_samples = X_test.shape[0]
            bin_y = np.zeros((y_pred_test.shape[0], n_targets), dtype='int64')
            for i in range(n_samples):
                bin_y[i, y_pred_test[i]] = 1
            y_pred_test = bin_y
            ys_pred_train.append(y_pred_train)
            ys_pred_test.append(y_pred_test)
        pred_df_train = transformer.inverse_transform(df_train, ys_pred_train)
        pred_df_test = transformer.inverse_transform(df_test, ys_pred_test)
    else:
        if model == 'trace':
            estimator = TraceNormEstimator(alpha=alpha,
                                           step_size_multiplier=1000,
                                           fit_intercept=True,
                                           max_backtracking_iter=10,
                                           momentum=True,
                                           beta=beta,
                                           max_iter=max_iter,
                                           verbose=verbose)
        elif model == 'non_convex':
            if source_init is not None:
                estimator = load(join(source_init, 'estimator.pkl'))
                info = json.load(open(join(source_init, 'info.json'), 'r'))
                n_components = info['rank']
                score = info['score']
                print('init', score)
                coef = estimator.coef_
                intercept = estimator.intercept_
            else:
                coef, intercept = None, None
            estimator = NonConvexEstimator(
                alpha=alpha, n_components=n_components,
                latent_dropout_rate=latent_dropout_rate,
                input_dropout_rate=input_dropout_rate,
                optimizer=optimizer,
                max_iter=max_iter,
                latent_sparsity=None,
                coef_init=coef,
                intercept_init=intercept,
                step_size=step_size)
        else:
            raise ValueError
        estimator.fit(Xs_train, ys_train)
        ys_pred_train = estimator.predict(Xs_train)
        pred_df_train = transformer.inverse_transform(df_train, ys_pred_train)
        ys_pred_test = estimator.predict(Xs_test)
        pred_df_test = transformer.inverse_transform(df_test, ys_pred_test)
    return pred_df_train, pred_df_test, estimator, transformer


@exp.automain
def main(datasets, source, reduced_dir, unmask_dir,
         n_subjects, test_size, train_size, model,
         _run, _seed):
    artifact_dir = join(_run.observers[0].basedir, str(_run._id))
    df = make_data_frame(datasets, source,
                         reduced_dir=reduced_dir,
                         unmask_dir=unmask_dir,
                         n_subjects=n_subjects)
    df_train, df_test = split_folds(df, test_size=test_size,
                                    train_size=train_size,
                                    random_state=_seed)

    pred_df_train, pred_df_test, estimator, transformer \
        = fit_model(df_train, df_test)

    pred_contrasts = pd.concat([pred_df_test, pred_df_train],
                               keys=['test', 'train'],
                               names=['fold'], axis=0)
    true_contrasts = pred_contrasts.index.get_level_values('contrast').values
    res = pd.DataFrame({'pred_contrast': pred_contrasts,
                        'true_contrast': true_contrasts})
    match = res['pred_contrast'] == res['true_contrast']
    score = match.groupby(level=['fold', 'dataset']).aggregate('mean')
    score_dict = {}
    for (fold, dataset), this_score in score.iteritems():
        score_dict['%s_%s' % (fold, dataset)] = this_score
    _run.info['score'] = score_dict

    if model in ['logistic', 'trace']:
        U, s, V = svd(estimator.coef_, full_matrices=False)
        rank = np.sum(s > 1e-7)
        dump(estimator, join(artifact_dir, 'estimator.pkl'))
    else:
        rank = estimator.n_components
    _run.info['rank'] = rank
    dump(transformer, join(artifact_dir, 'transformer.pkl'))
    print('rank', rank)
    print(score)
