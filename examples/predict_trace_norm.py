from os.path import join

import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver

from cogspaces.trace_norm import TraceNormEstimator
from cogspaces.utils import get_output_dir, make_data_frame, split_folds, \
    MultiDatasetTransformer

idx = pd.IndexSlice

exp = Experiment('Clean')
basedir = join(get_output_dir(), 'clean')
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    datasets = ['brainomics']
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')
    source = 'hcp_rs'
    n_subjects = None
    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5}
    train_size = {'hcp': .9, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                  'la5c': .5}
    alpha = 0
    beta = 0
    n_iter = 3000
    verbose = 10
    seed = 10


@exp.automain
def main(datasets, source, reduced_dir, unmask_dir,
         n_subjects, test_size, train_size, n_iter, alpha, beta,
         verbose,
         _seed):
    df = make_data_frame(datasets, source,
                         reduced_dir=reduced_dir,
                         unmask_dir=unmask_dir,
                         n_subjects=n_subjects)
    df_train, df_test = split_folds(df, test_size=test_size,
                                    train_size=train_size,
                                    random_state=_seed)
    transformer = MultiDatasetTransformer()
    Xs_train, ys_train = transformer.fit_transform(df_train)
    Xs_test, ys_test = transformer.fit_transform(df_test)

    estimator = TraceNormEstimator(alpha=alpha,
                                   step_size_multiplier=1,
                                   beta=beta,
                                   n_iter=n_iter,
                                   verbose=verbose)
    estimator.fit(Xs_train, ys_train)
    ys_pred_train = estimator.predict(Xs_train)
    pred_df_train = transformer.inverse_transform(df_train, ys_pred_train)
    ys_pred_test = estimator.predict(Xs_test)
    pred_df_test = transformer.inverse_transform(df_test, ys_pred_test)

    pred_contrasts = pd.concat([pred_df_test, pred_df_train],
                               keys=['test', 'train'],
                               names=['fold'], axis=0)
    true_contrasts = pred_contrasts.index.get_level_values('contrast').values
    res = pd.DataFrame({'pred_contrast': pred_contrasts,
                        'true_contrast': true_contrasts})
    match = res['pred_contrast'] == res['true_contrast']
    score = match.groupby(level=['fold', 'dataset']).aggregate('mean')
    print(score)
