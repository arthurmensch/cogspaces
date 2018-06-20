# Load data


import numpy as np
import os
from joblib import dump, Memory
from matplotlib.testing.compare import get_cache_dir
from os.path import join
from sacred import Experiment
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.model_selection import train_test_split
from cogspaces.models.factored import FactoredClassifier
from cogspaces.models.factored_dl import FactoredDL
from cogspaces.models.factored_ss import StudySelector
from cogspaces.models.logistic import MultiLogisticClassifier
from cogspaces.preprocessing import MultiStandardScaler, MultiTargetEncoder
from cogspaces.utils.callbacks import ScoreCallback, MultiCallback
from cogspaces.utils.sacred import OurFileStorageObserver

exp = Experiment('multi_studies')


@exp.config
def default():
    seed = 860
    full = False
    system = dict(
        device=-1,
        verbose=2,
        n_jobs=3,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512_gm'),
        studies=['brainomics', 'archi'],
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
        refinement=None,
        target_study='archi',
    )
    factored = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        black_list_target=True,
        regularization=1,
        adaptive_dropout=True,
        sampling='random',
        weight_power=0.6,
        batch_size=128,
        epoch_counting='all',
        init='normal',
        batch_norm=True,
        reset_classifiers=False,
        # refit_from=join(get_output_dir(), 'factored_gm',
        #                 'dl_rest_860_1e-04.pkl'),
        dropout=0.5,
        input_dropout=0.25,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4,
            'finetune': 1e-3},
        max_iter={'pretrain': 0, 'train': 300, 'sparsify': 0,
                  'finetune': 200},
        refit_data=['classifier', 'dropout']
    )

    logistic = dict(
        estimator='logistic',
        l2_penalty=[7e-5],
        max_iter=3000,
        refit_from=None,
    )

    refinement = dict(
        n_runs=45,
        n_splits=3,
        alpha=1e-3,
        warmup=False
    )


@exp.named_config
def dl():
    seed = 10
    full = True
    system = dict(
        device=-1,
        verbose=2,
        n_jobs=45,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='all',
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
        refinement='dl',
        target_study=None,
    )
    factored = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        adaptive_dropout=True,
        sampling='random',
        weight_power=0.6,
        batch_size=128,
        epoch_counting='all',
        init='rest',
        batch_norm=True,
        dropout=0.5,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4,
            'finetune': 1e-3},
        input_dropout=0.25,
        max_iter={'pretrain': 200, 'train': 300, 'sparsify': 0,
                  'finetune': 200},
    )

    logistic = dict(
        estimator='logistic',
        l2_penalty=np.logspace(-5, 1, 7).tolist(),
        max_iter=1000,
        refit_from=None
    )

    refinement = dict(
        n_runs=135,
        alpha=1e-3,
        warmup=True
    )


@exp.named_config
def ss():
    seed = 10
    full = False
    system = dict(
        device=-1,
        verbose=2,
        n_jobs=3,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='all',
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
        refinement='study_selector',
        target_study='ds009',
    )
    factored = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        adaptive_dropout=True,
        sampling='random',
        weight_power=0.6,
        batch_size=32,
        epoch_counting='all',
        init='rest',
        batch_norm=True,
        # full_init=join(get_output_dir(), 'seed_split_init', 'pca_15795.pkl'),
        dropout=0.75,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4,
            'finetune': 1e-3},
        input_dropout=0.25,
        max_iter={'pretrain': 2, 'train': 3, 'sparsify': 0,
                  'finetune': 2},
    )

    logistic = dict(
        estimator='logistic',
        l2_penalty=np.logspace(-5, 1, 7).tolist(),
        max_iter=1000,
        refit_from=None
    )

    refinement = dict(
        n_runs=3,
        n_splits=3
    )


@exp.capture
def save_output(target_encoder, standard_scaler, estimator, scores, _run):
    if not _run.unobserved:
        try:
            observer = next(filter(lambda x:
                                   isinstance(x, OurFileStorageObserver),
                                   _run.observers))
        except StopIteration:
            return
        dump(target_encoder, join(observer.dir, 'target_encoder.pkl'))
        dump(standard_scaler, join(observer.dir, 'standard_scaler.pkl'))
        dump(estimator, join(observer.dir, 'estimator.pkl'))
        dump(scores, join(observer.dir, 'scores.pkl'))


@exp.capture(prefix='data')
def load_data(source_dir, studies):
    data, target = load_data_from_dir(data_dir=source_dir)
    for study, this_target in target.items():
        this_target['all_contrast'] = study + '_' + this_target['contrast']

    if studies == 'all':
        studies = list(sorted(data.keys()))
    elif isinstance(studies, str):
        studies = [studies]
    elif not isinstance(studies, list):
        raise ValueError("Studies should be a list or 'all'")
    data = {study: data[study] for study in studies}
    target = {study: target[study] for study in studies}
    return data, target


@exp.main
def train(system, model, logistic, refinement,
          factored, full,
          _run, _seed):
    data, target = load_data()
    print(_seed)

    target_encoder = MultiTargetEncoder().fit(target)
    target = target_encoder.transform(target)

    train_data, test_data, train_targets, test_targets = \
        train_test_split(data, target, random_state=_seed)
    if full:
        train_data = data
        train_targets = target

    if model['normalize']:
        standard_scaler = MultiStandardScaler().fit(train_data)
        train_data = standard_scaler.transform(train_data)
        test_data = standard_scaler.transform(test_data)
    else:
        standard_scaler = None

    if model['estimator'] == 'factored':
        estimator = FactoredClassifier(verbose=system['verbose'],
                                       device=system['device'],
                                       target_study=model['target_study'],
                                       n_jobs=system['n_jobs'],
                                       **factored)
        if model['target_study'] is not None:
            target_study = model['target_study']
            test_data = {target_study: test_data[target_study]}
            test_targets = {target_study: test_targets[target_study]}
            train_test_data = {target_study: train_data[target_study]}
            train_test_targets = {target_study: train_targets[target_study]}
        else:
            train_test_data = train_data
            train_test_targets = train_targets
        if model['refinement'] == 'ss':
            if model['target_study'] is None:
                raise ValueError("Refinement 'study_selector' requires"
                                 " 'target_study' to be set.")
            estimator = StudySelector(estimator, model['target_study'],
                                      n_jobs=system['n_jobs'],
                                      n_runs=refinement['n_runs'],
                                      n_splits=refinement['n_splits'],
                                      seed=factored['seed'])
        elif model['refinement'] == 'dl':
            estimator = FactoredDL(estimator,
                                   n_jobs=system['n_jobs'],
                                   n_runs=refinement['n_runs'],
                                   alpha=refinement['alpha'],
                                   warmup=refinement['warmup'],
                                   memory=Memory(cachedir=get_cache_dir()),
                                   seed=factored['seed'])
        elif model['refinement'] is not None:
            raise ValueError('Wrong parameter for `refinement`: %s' %
                             model['refinement'])
    elif model['estimator'] == 'logistic':
        estimator = MultiLogisticClassifier(verbose=system['verbose'],
                                            **logistic)
        train_test_data = train_data
        train_test_targets = train_targets
    else:
        return ValueError("Wrong value for parameter "
                          "`model.estimator`: got '%s'."
                          % model['estimator'])
    test_callback = ScoreCallback(X=test_data, y=test_targets,
                                  score_function=accuracy_score)
    train_callback = ScoreCallback(X=train_test_data, y=train_test_targets,
                                   score_function=accuracy_score)
    callback = MultiCallback({'train': train_callback,
                              'test': test_callback})
    _run.info['n_iter'] = train_callback.n_iter_
    _run.info['train_scores'] = train_callback.scores_
    _run.info['test_scores'] = test_callback.scores_

    estimator.fit(train_data, train_targets, callback=callback)

    if hasattr(estimator, 'dropout_'):
        _run.info['dropout'] = estimator.dropout_
    if hasattr(estimator, 'studies_'):
        _run.info['studies'] = estimator.studies_

    test_preds = estimator.predict(test_data)

    test_scores = {}
    all_f1 = {}
    all_prec = {}
    all_recall = {}
    all_confusion = {}
    for study in test_preds:
        these_preds = test_preds[study]['contrast']
        these_targets = test_targets[study]['contrast']
        test_scores[study] = accuracy_score(these_preds,
                                            these_targets)
        precs, recalls, f1s, support = precision_recall_fscore_support(
            these_preds, these_targets, warn_for=())
        contrasts = target_encoder.le_[study]['contrast'].classes_
        all_prec[study] = {contrast: prec for contrast, prec in
                           zip(contrasts, precs)}
        all_recall[study] = {contrast: recall for contrast, recall in
                             zip(contrasts, recalls)}
        all_f1[study] = {contrast: f1 for contrast, f1 in
                          zip(contrasts, f1s)}
        all_confusion[study] = confusion_matrix(these_preds, these_targets)
    scores = (all_confusion, all_prec, all_recall, all_f1)
    save_output(target_encoder, standard_scaler, estimator, scores)

    return test_scores


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'multi_studies')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    exp.observers.append(OurFileStorageObserver.create(basedir=output_dir))
    run = exp.run_commandline()