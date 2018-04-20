# Load data
from math import sqrt

import numpy as np
import pandas as pd
from joblib import dump
from os.path import join
from sacred import Experiment
from sklearn.metrics import accuracy_score

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.model_selection import train_test_split
from cogspaces.models.baseline import MultiLogisticClassifier
from cogspaces.models.factored import FactoredClassifier, FactoredClassifierCV
from cogspaces.models.factored_fast import MultiStudyClassifier
from cogspaces.models.trace import TraceClassifier
from cogspaces.models.variational import VarMultiStudyClassifier
from cogspaces.preprocessing import MultiStandardScaler, MultiTargetEncoder
from cogspaces.utils.callbacks import ScoreCallback, MultiCallback
from cogspaces.utils.sacred import OurFileStorageObserver

exp = Experiment('multi_studies')


@exp.config
def default():
    seed = 10
    system = dict(
        device=-1,
        verbose=10,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='all',
        target_study='archi'
    )
    model = dict(
        normalize=False,
        estimator='factored_fast',
        study_weight='sqrt_sample',
        max_iter=200,
    )
    factored_fast = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        epoch_counting='all',
        sampling='random',
        batch_size=128,
        regularization=1e-4,
        dropout=0.75,
        lr=1e-3,
        input_dropout=0.25)

    factored_variational = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        epoch_counting='all',
        sampling='random',
        batch_size=128,
        regularization=.05,
        dropout=0.75,
        lr=1e-3,
        input_dropout=0.25)

    factored = dict(
        optimizer='adam',
        adapt_size=0,
        shared_latent_size=64,
        private_latent_size=0,
        shared_latent='hard',
        skip_connection=False,
        activation='linear',
        epoch_counting='all',
        sampling='cycle',
        fine_tune=True,
        decode=False,
        batch_size=128,
        dropout=0.75,
        loss_weights={'contrast': 1, 'study': 1, 'penalty': 1,
                      'decoding': 1, 'all_contrast': 1},
        lr=1e-3,
        batch_norm=True,
        input_dropout=0.25)
    factored_cv = dict(
        optimizer='adam',
        shared_latent_size=10,
        private_latent_size=0,
        averaging=False,
        shared_latent='hard',
        skip_connection=False,
        activation='linear',
        decode=False,
        sampling='weighted_random',
        batch_size=128,
        dropout=0.,
        n_jobs=20,
        n_splits=10,
        n_runs=1,
        lr=1e-3,
        input_dropout=0.25)
    trace = dict(
        trace_penalty=1e-3,
    )
    logistic = dict(
        l2_penalty=1e-3,
    )


@exp.capture
def save_output(target_encoder, standard_scaler, estimator,
                test_preds, test_latents, train_latents, _run):
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
        dump(test_preds, join(observer.dir, 'test_preds.pkl'))
        dump(test_latents, join(observer.dir, 'test_latents.pkl'))
        dump(train_latents, join(observer.dir, 'train_latents.pkl'))


@exp.capture(prefix='data')
def load_data(source_dir, studies, target_study):
    data, target = load_data_from_dir(data_dir=source_dir)
    for study in target:
        target[study]['all_contrast'] = target[study]['study'] \
                                        + '_' + target[study]['contrast']

    if studies == 'all':
        studies = list(data.keys())
    elif isinstance(studies, str):
        studies = [studies]
    elif not isinstance(studies, list):
        raise ValueError("Studies should be a list or 'all'")
    studies.remove(target_study)
    studies = [target_study] + studies
    data = {study: data[study] for study in studies}
    target = {study: target[study] for study in studies}
    return data, target


@exp.main
def train(system, model, factored, factored_cv, trace, logistic,
          factored_fast, factored_variational,
          _run, _seed):
    print(_seed)
    data, target = load_data()

    target_encoder = MultiTargetEncoder().fit(target)
    target = target_encoder.transform(target)

    train_data, test_data, train_targets, test_targets = \
        train_test_split(data, target, random_state=_seed)

    if model['normalize']:
        standard_scaler = MultiStandardScaler().fit(train_data)
        train_data = standard_scaler.transform(train_data)
        test_data = standard_scaler.transform(test_data)
    else:
        standard_scaler = None

    study_weights = get_study_weights(model['study_weight'], train_data)

    if model['estimator'] == 'factored':
        estimator = FactoredClassifier(verbose=system['verbose'],
                                       device=system['device'],
                                       max_iter=model['max_iter'],
                                       seed=_seed,
                                       **factored)
    elif model['estimator'] == 'factored_cv':
        estimator = FactoredClassifierCV(verbose=system['verbose'],
                                         device=system['device'],
                                         max_iter=model['max_iter'],
                                         seed=_seed,
                                         **factored_cv)
    elif model['estimator'] == 'factored_fast':
        estimator = MultiStudyClassifier(verbose=system['verbose'],
                                            device=system['device'],
                                            max_iter=model['max_iter'],
                                            seed=_seed,
                                            **factored_fast)
    elif model['estimator'] == 'factored_variational':
        estimator = VarMultiStudyClassifier(verbose=system['verbose'],
                                            device=system['device'],
                                            max_iter=model['max_iter'],
                                            seed=_seed,
                                            **factored_variational)
    elif model['estimator'] == 'trace':
        estimator = TraceClassifier(verbose=system['verbose'],
                                    max_iter=model['max_iter'],
                                    step_size_multiplier=100000,
                                    **trace)
    elif model['estimator'] == 'logistic':
        estimator = MultiLogisticClassifier(verbose=system['verbose'],
                                            max_iter=model['max_iter'],
                                            **logistic)
    else:
        return ValueError("Wrong value for parameter "
                          "`model.estimator`: got '%s'."
                          % model['estimator'])
    test_callback = ScoreCallback(X=test_data, y=test_targets,
                                  score_function=accuracy_score)
    train_callback = ScoreCallback(X=train_data, y=train_targets,
                                   score_function=accuracy_score)
    callback = MultiCallback({'train': train_callback,
                              'test': test_callback})
    _run.info['n_iter'] = train_callback.n_iter_
    _run.info['train_scores'] = train_callback.scores_
    _run.info['test_scores'] = test_callback.scores_

    estimator.fit(train_data, train_targets,
                  study_weights=study_weights,
                  callback=callback
                  )

    if model['estimator'] == 'factored_cv':
        target = list(test_data.keys())[0]
        test_data = {target: test_data[target]}

    if hasattr(estimator, 'studies_'):
        _run.info['studies'] = estimator.studies_

    test_preds = estimator.predict(test_data)

    test_latents = estimator.predict_latent(test_data)
    train_latents = estimator.predict_latent(train_data)

    test_scores = {}
    for study in test_preds:
        test_scores[study] = accuracy_score(test_preds[study]['contrast'],
                                            test_targets[study]['contrast'])

    test_preds = target_encoder.inverse_transform(test_preds)
    test_targets = target_encoder.inverse_transform(test_targets)
    for study in test_preds:
        test_preds[study] = pd.concat([test_preds[study], test_targets[study]],
                                      axis=1,
                                      keys=['pred', 'true'], names=['target'])
    save_output(target_encoder, standard_scaler, estimator, test_preds,
                test_latents, train_latents)
    return test_scores


def get_study_weights(study_weight, train_data):
    if study_weight == 'sqrt_sample':
        study_weights = np.array(
            [sqrt(len(train_data[study])) for study in train_data])
        s = np.sum(study_weights)
        study_weights /= s / len(train_data)
        study_weights = {study: weight for study, weight in zip(train_data,
                                                                study_weights)}
    elif study_weight == 'sample':
        study_weights = np.array(
            [float(len(train_data[study])) for study in train_data])
        s = float(np.sum(study_weights))
        study_weights /= s / len(train_data)
        study_weights = {study: weight for study, weight in zip(train_data,
                                                                study_weights)}
    elif study_weight == 'study':
        study_weights = {study: 1. for study in train_data}
    elif study_weight == 'target':
        study_weights = {study: 1. if i == 0 else 1. / (len(train_data) - 1)
                         for i, study in enumerate(train_data)}
    else:
        raise ValueError
    return study_weights


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'multi_studies')
    exp.observers.append(OurFileStorageObserver.create(basedir=output_dir))
    exp.run()
