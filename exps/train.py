# Load data
from os.path import join

from joblib import dump
from sacred import Experiment
from sklearn.metrics import accuracy_score

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.model_selection import train_test_split
from cogspaces.models.baseline import MultiLogisticClassifier
from cogspaces.models.factored import FactoredClassifier
from cogspaces.models.trace import TraceClassifier
from cogspaces.preprocessing import MultiStandardScaler, MultiTargetEncoder
from cogspaces.utils.callbacks import ScoreCallback
from cogspaces.utils.sacred import OurFileStorageObserver

exp = Experiment('multi_studies')
output_dir = join(get_output_dir(), 'multi_studies')
exp.observers.append(OurFileStorageObserver.create(basedir=output_dir))


@exp.config
def default():
    system = dict(
        device=-1,
        seed=0,
        verbose=100,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies=['archi', 'hcp', 'brainomics']
    )
    model = dict(
        normalize=True,
        estimator='trace',
        max_iter=1000,
    )
    factored = dict(
        optimizer='adam',
        embedding_size=20,
        batch_size=128,
        dropout=0.,
        input_dropout=0.0,
        l2_penalty=0,
    )
    trace = dict(
        trace_penalty=5e-2,
    )
    logistic = dict(
        l2_penalty=1e-3,
    )


@exp.capture
def save_output(target_encoder, standard_scaler, estimator,
                test_preds, _run):
    if not _run.unobserved:
        try:
            observer = next(filter(lambda x: isinstance(x, OurFileStorageObserver),
                                   _run.observers))
        except StopIteration:
            return
        dump(target_encoder, join(observer.dir, 'target_encoder.pkl'))
        dump(standard_scaler, join(observer.dir, 'standard_scaler.pkl'))
        dump(estimator, join(observer.dir, 'estimator.pkl'))
        dump(test_preds, join(observer.dir, 'test_preds.pkl'))


@exp.capture(prefix='data')
def load_data(source_dir, studies):
    data, target = load_data_from_dir(data_dir=source_dir)

    if studies == 'all':
        studies = list(data.keys())
    elif isinstance(studies, str):
        studies = [studies]
    elif not isinstance(studies, list):
        raise ValueError("Studies should be a list or 'all'")
    data = {study: data[study] for study in studies}
    target = {study: target[study] for study in studies}
    return data, target


@exp.automain
def train(system, model, factored, trace, logistic,
          _run, _seed):
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

    train_contrasts = {study: train_target['contrast'] for study, train_target
                       in train_targets.items()}
    test_contrasts = {study: test_target['contrast'] for study, test_target
                      in test_targets.items()}

    if model['estimator'] == 'factored':
        estimator = FactoredClassifier(verbose=system['verbose'],
                                       device=system['device'],
                                       max_iter=model['max_iter'],
                                       **factored)
    elif model['estimator'] == 'trace':
        estimator = TraceClassifier(verbose=system['verbose'],
                                    max_iter=model['max_iter'],
                                    step_size_multiplier=10000,
                                    **trace)
    elif model['estimator'] == 'logistic':
        estimator = MultiLogisticClassifier(verbose=system['verbose'],
                                            max_iter=model['max_iter'],
                                            **logistic)
    else:
        return ValueError("Wrong value for parameter `model.estimator`: got '%s'."
                          % model['estimator'])

    callback = ScoreCallback(estimator, X=test_data, y=test_contrasts,
                             score_function=accuracy_score)
    _run.info['n_iter'] = callback.n_iter_
    _run.info['scores'] = callback.scores_

    estimator.fit(train_data, train_contrasts, callback=callback)

    test_pred_contrasts = estimator.predict(test_data)
    test_scores = {}
    for study in test_contrasts:
        test_scores[study] = accuracy_score(test_pred_contrasts[study],
                                            test_contrasts[study])

    test_preds = {}
    for study, test_target in test_targets.items():
        test_preds[study] = test_target.copy()
        test_preds[study]['contrast'] = test_pred_contrasts[study]
    test_preds = target_encoder.inverse_transform(test_preds)
    for study in test_preds:
        test_preds[study]['true_contrast'] = test_targets[study]['contrast']

    save_output(target_encoder, standard_scaler, estimator, test_preds)
    return test_scores
