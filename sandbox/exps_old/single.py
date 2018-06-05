import pandas as pd
import torch
from cogspaces.model.non_convex_pytorch import TransferEstimator
from cogspaces.pipeline import get_output_dir, make_data_frame, split_folds, \
    MultiDatasetTransformer
from joblib import dump
from os.path import join
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from cogspaces.models.trace import TransferTraceNormEstimator

idx = pd.IndexSlice

exp = Experiment('single_exp')
basedir = join(get_output_dir(), 'single_exp')
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    datasets = ['brainomics', 'hcp']
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')
    # source = 'mix'
    source = 'hcp_new'
    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5, 'full': .5}
    train_size = dict(hcp=None, archi=None, la5c=None, brainomics=None,
                      camcan=None,
                      human_voice=None)
    dataset_weights = {'brainomics': 1, 'archi': 1, 'hcp': 1}
    model = 'trace'
    max_iter = 500
    verbose = 10
    seed = 100

    with_std = True
    with_mean = True
    per_dataset = True

    # Factored only
    n_components = 10

    batch_size = 128
    optimizer = 'adam'
    step_size = 1e-3

    alphas = [5e-4]  # np.logspace(-6, -1, 12)
    latent_dropout_rates = [0.0]
    input_dropout_rates = [0.0]
    dataset_weights_helpers = [[1.]]

    n_splits = 10
    n_jobs = 1


@exp.capture
def fit_model(df_train, df_test, model,
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
                                          integer_coding=model != 'trace',
                                          per_dataset=per_dataset)
    transformer.fit(df_train)
    Xs_train, ys_train = transformer.transform(df_train)

    Xs_test, ys_test = transformer.transform(df_test)
    X_train, Xs_train_helpers = Xs_train[0], Xs_train[1:]
    y_train, ys_train_helpers = ys_train[0], ys_train[1:]
    X_test = Xs_test[0]

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25)

    cross_val = len(alphas) > 1

    if model == 'logistic_l2':
        torch.set_num_threads(1)
        estimator = TransferEstimator(
            n_components=n_components,
            architecture='flat',
            batch_size=batch_size,
            optimizer=optimizer,
            alpha=alphas[0],
            max_iter=max_iter,
            step_size=step_size, n_jobs=1)
        if cross_val:
            estimator = GridSearchCV(estimator,
                                     cv=cv,
                                     n_jobs=n_jobs,
                                     param_grid={'alpha': alphas})
    elif model == 'logistic_dropout':
        torch.set_num_threads(1)
        estimator = TransferEstimator(
            alpha=0.,
            n_components=n_components,
            architecture='flat',
            batch_size=batch_size,
            optimizer=optimizer,
            max_iter=max_iter,
            input_dropout_rate=input_dropout_rates[0],
            step_size=step_size, n_jobs=1)
        cross_val = len(input_dropout_rates) > 1
        if cross_val:
            estimator = GridSearchCV(estimator,
                                     cv=cv,
                                     n_jobs=n_jobs,
                                     param_grid={
                                         'input_dropout_rate': input_dropout_rates})
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
            latent_dropout_rate=latent_dropout_rates[0],
            input_dropout_rate=input_dropout_rates[0],
            dataset_weights_helpers=dataset_weights_helpers[0],
            ys_helpers=ys_train_helpers, )
        cross_val = len(input_dropout_rates) > 1 \
                    or len(latent_dropout_rates) > 1 \
                    or len(dataset_weights_helpers) > 1
        if cross_val:
            estimator = GridSearchCV(estimator,
                                     n_jobs=n_jobs,
                                     cv=cv,
                                     param_grid={
                                         'latent_dropout_rate': latent_dropout_rates,
                                         'input_dropout_rate': input_dropout_rates,
                                         'dataset_weights_helpers':
                                             dataset_weights_helpers})
    elif model == 'logistic_l2_sklearn':
        n_samples = X_train.shape[0]
        if cross_val:
            estimator = LogisticRegressionCV(solver='lbfgs', max_iter=max_iter,
                                             Cs=1. / alphas / n_samples,
                                             n_jobs=n_jobs,
                                             cv=cv,
                                             verbose=10)
        else:
            alpha = alphas[0]
            estimator = LogisticRegression(solver='lbfgs', max_iter=max_iter,
                                           multi_class='multinomial',
                                           C=1. / alpha / n_samples,
                                           n_jobs=n_jobs,
                                           verbose=10)
    elif model == 'trace':
        alpha = alphas[0]
        estimator = TransferTraceNormEstimator(alpha=alpha,
                                               step_size_multiplier=1000,
                                               fit_intercept=True,
                                               Xs_helpers=Xs_train_helpers,
                                               ys_helpers=ys_train_helpers,
                                               dataset_weights_helpers=
                                               dataset_weights_helpers[0],
                                               max_backtracking_iter=10,
                                               momentum=True,
                                               split_loss=True,
                                               beta=0,
                                               max_iter=max_iter,
                                               verbose=10)
    else:
        raise ValueError('Wrong model argument')

    estimator.fit(X_train, y_train)

    if model != 'logistic_l2_sklearn':
        if cross_val:
            ys_pred_train = estimator.best_estimator_.predict_all(Xs_train)
            ys_pred_test = estimator.best_estimator_.predict_all(Xs_test)
        else:
            ys_pred_train = estimator.predict_all(Xs_train)
            ys_pred_test = estimator.predict_all(Xs_test)
        pred_df_train = transformer.inverse_transform(df_train, ys_pred_train)
        pred_df_test = transformer.inverse_transform(df_test, ys_pred_test)
    else:
        y_pred_train = estimator.predict(X_train)
        y_pred_test = estimator.predict(X_test)
        pred_df_train = transformer.inverse_transform(df_train, [y_pred_train])
        pred_df_test = transformer.inverse_transform(df_test, [y_pred_test])

    return pred_df_train, pred_df_test, estimator, transformer


@exp.automain
def main(datasets, source, reduced_dir, unmask_dir,
         test_size, train_size,
         _run, _seed):
    artifact_dir = join(_run.observers[0].basedir, str(_run._id))
    if source in ['hcp_rs_positive_single', 'hcp_new_big_single']:
        source = 'hcp_new_big'
        indices = slice(-512, None)
    elif source in ['hcp_new_single']:
        source = 'hcp_new'
        indices = slice(-128, None)
    else:
        indices = slice(None)
    df = make_data_frame(datasets, source,
                         reduced_dir=reduced_dir,
                         unmask_dir=unmask_dir)
    df = df.iloc[:, indices]
    df_train, df_test = split_folds(df, test_size=test_size,
                                    train_size=train_size,
                                    random_state=_seed)
    pred_df_train, pred_df_test, estimator, transformer \
        = fit_model(df_train, df_test, )

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
