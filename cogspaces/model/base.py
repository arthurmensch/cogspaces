import json
from os.path import join

import numpy as np
from sklearn.externals.joblib import load
from sklearn.linear_model import LogisticRegression

from .convex import TraceNormEstimator
from .non_convex import NonConvexEstimator
from ..pipeline import MultiDatasetTransformer, get_output_dir


def fit_model(df_train, df_test, model, alpha, beta, n_components,
              max_iter, verbose):
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
                                           solver='lbfgs',
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
            # source_init = join(get_output_dir(), 'clean', '557')
            # estimator = load(join(source_init, 'estimator.pkl'))
            # info = json.load(open(join(source_init, 'info.json'), 'r'))
            # n_components = info['rank']
            # score = info['score']
            # print('init', score)
            # coef = estimator.coef_
            # intercept = estimator.intercept_
            estimator = NonConvexEstimator(alpha=1e-3,
                                           n_components=n_components,
                                           latent_dropout_rate=0.,
                                           input_dropout_rate=0.,
                                           optimizer='sgd',
                                           max_iter=max_iter,
                                           latent_sparsity=None,
                                           # coef_init=coef,
                                           # intercept_init=intercept,
                                           step_size=10)
        else:
            raise ValueError
        estimator.fit(Xs_train, ys_train)
        ys_pred_train = estimator.predict(Xs_train)
        pred_df_train = transformer.inverse_transform(df_train, ys_pred_train)
        ys_pred_test = estimator.predict(Xs_test)
        pred_df_test = transformer.inverse_transform(df_test, ys_pred_test)
    return pred_df_train, pred_df_test, estimator, transformer