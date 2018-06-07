# Load data

from joblib import dump
from os.path import join
from sacred import Experiment
from sklearn.metrics import accuracy_score

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.model_selection import train_test_split
from cogspaces.models.baseline import MultiLogisticClassifier
from cogspaces.models.variational import VarMultiStudyClassifier
from cogspaces.preprocessing import MultiStandardScaler, MultiTargetEncoder
from cogspaces.utils.callbacks import ScoreCallback, MultiCallback
from cogspaces.utils.sacred import OurFileStorageObserver

exp = Experiment('multi_studies')


@exp.config
def default():
    seed = 10
    full = False
    system = dict(
        device=-1,
        verbose=2,
    )
    data = dict(
        source_dir=join(get_data_dir(), 'reduced_512'),
        studies='all',
    )
    model = dict(
        estimator='factored',
        normalize=False,
        seed=100,
    )
    factored = dict(
        optimizer='adam',
        latent_size=128,
        activation='linear',
        regularization=1,
        epoch_counting='all',
        adaptive_dropout=False,
        sampling='random',
        weight_power=0.6,
        batch_size=128,
        init='symmetric',
        finetune_dropouts=None,
        dropout=0.5,
        lr=1e-3,
        input_dropout=0.25,
        max_iter={'pretrain': 10, 'sparsify': 0, 'finetune': 10},
    )

    logistic = dict(
        l2_penalty=1e-5,
        max_iter=200,
        reduction=None
    )


@exp.capture
def save_output(target_encoder, standard_scaler, estimator, _run):
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
def train(system, model, logistic,
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
        estimator = VarMultiStudyClassifier(verbose=system['verbose'],
                                            device=system['device'],
                                            **factored)
    elif model['estimator'] == 'logistic':
        estimator = MultiLogisticClassifier(verbose=system['verbose'],
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

    estimator.fit(train_data, train_targets, callback=callback)

    if hasattr(estimator, 'dropout_'):
        _run.info['dropout'] = estimator.dropout_

    test_preds = estimator.predict(test_data)

    test_scores = {}
    for study in test_preds:
        test_scores[study] = accuracy_score(test_preds[study]['contrast'],
                                            test_targets[study]['contrast'])
    save_output(target_encoder, standard_scaler, estimator)

    return test_scores


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'multi_studies')
    exp.observers.append(OurFileStorageObserver.create(basedir=output_dir))
    run = exp.run()
    output_dir = join(output_dir, str(run._id))
