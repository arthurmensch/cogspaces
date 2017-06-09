from os.path import join

import numpy as np
import pandas as pd
from cogspaces.model import fit_model
from cogspaces.pipeline import get_output_dir, make_data_frame, split_folds
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.externals.joblib import dump

idx = pd.IndexSlice

exp = Experiment('Predict')
basedir = join(get_output_dir(), 'predict')
exp.observers.append(FileStorageObserver.create(basedir=basedir))


@exp.config
def config():
    datasets = ['hcp', 'archi', 'brainomics']
    reduced_dir = join(get_output_dir(), 'reduced')
    unmask_dir = join(get_output_dir(), 'unmasked')
    source = 'hcp_rs_positive'
    n_subjects = None
    test_size = {'hcp': .1, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                 'la5c': .5}
    train_size = {'hcp': .9, 'archi': .5, 'brainomics': .5, 'camcan': .5,
                  'la5c': .5}
    model = 'non_convex'
    alpha = 1e-3
    beta = 1e-5
    n_components = 50
    max_iter = 1000
    verbose = 10
    seed = 10


@exp.automain
def main(datasets, source, reduced_dir, unmask_dir,
         n_subjects, test_size, train_size, max_iter, alpha, beta,
         n_components, model,
         verbose, _run, _seed):
    artifact_dir = join(_run.observers[0].basedir, str(_run._id))
    df = make_data_frame(datasets, source,
                         reduced_dir=reduced_dir,
                         unmask_dir=unmask_dir,
                         n_subjects=n_subjects)
    df_train, df_test = split_folds(df, test_size=test_size,
                                    train_size=train_size,
                                    random_state=_seed)

    pred_df_train, pred_df_test, estimator, transformer\
        = fit_model(df_train, df_test, model,
                    alpha, beta, n_components, max_iter, verbose)

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
        rank = np.linalg.matrix_rank(estimator.coef_)
        dump(estimator, join(artifact_dir, 'estimator.pkl'))
    else:
        rank = estimator.n_components
    _run.info['rank'] = rank
    dump(transformer, join(artifact_dir, 'transformer.pkl'))
    print('rank', rank)
    print(score)
