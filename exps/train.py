# Load data
import json
import os
from os.path import join, expanduser

from joblib import dump, Memory
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support

from cogspaces.classification.factored import FactoredClassifier
from cogspaces.classification.factored_dl import FactoredDL
from cogspaces.classification.logistic import MultiLogisticClassifier
from cogspaces.datasets import load_reduced_loadings, STUDY_LIST
from cogspaces.model_selection import train_test_split
from cogspaces.preprocessing import MultiStandardScaler, MultiTargetEncoder
from cogspaces.utils.callbacks import ScoreCallback, MultiCallback


def save_output(target_encoder, standard_scaler, estimator, scores, info):
    dump(target_encoder, join(output_dir, 'target_encoder.pkl'))
    dump(standard_scaler, join(output_dir, 'standard_scaler.pkl'))
    dump(estimator, join(output_dir, 'estimator.pkl'))
    dump(scores, join(output_dir, 'scores.pkl'))
    with open(join(output_dir, 'info.json'), 'w+') as f:
        json.dump(info, f)


def load_data(studies, data_dir=None):
    data, target = load_reduced_loadings(data_dir=data_dir)

    if studies == 'all':
        studies = STUDY_LIST
    elif isinstance(studies, str):
        studies = [studies]
    elif not isinstance(studies, list):
        raise ValueError("Studies should be a list or 'all'")
    data = {study: data[study] for study in studies}
    target = {study: target[study] for study in studies}
    return data, target


def train(data, system, model, factored, logistic, ensemble):
    input_data, target = load_data(data['studies'])

    target_encoder = MultiTargetEncoder().fit(target)
    target = target_encoder.transform(target)

    train_data, test_data, train_targets, test_targets = \
        train_test_split(input_data, target, random_state=system['seed'],
                         test_size=data['test_size'],
                         train_size=data['train_size'])

    info = {}
    if model['normalize']:
        standard_scaler = MultiStandardScaler().fit(train_data)
        train_data = standard_scaler.transform(train_data)
        test_data = standard_scaler.transform(test_data)
    else:
        standard_scaler = None

    if model['estimator'] == 'factored':
        ensemble = factored.pop('ensemble')
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
        if ensemble:
            estimator = FactoredDL(estimator,
                                   n_jobs=system['n_jobs'],
                                   n_runs=ensemble['n_runs'],
                                   alpha=ensemble['alpha'],
                                   warmup=ensemble['warmup'],
                                   memory=Memory(cachedir=expanduser('~/cache')),
                                   seed=factored['seed'])
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
    info['n_iter'] = train_callback.n_iter_
    info['train_scores'] = train_callback.scores_
    info['test_scores'] = test_callback.scores_

    estimator.fit(train_data, train_targets, callback=callback)

    if hasattr(estimator, 'dropout_'):
        info['dropout'] = estimator.dropout_
    if hasattr(estimator, 'studies_'):
        info['studies'] = estimator.studies_

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
    info['test_accuracy'] = test_scores
    save_output(target_encoder, standard_scaler, estimator, scores, info)
    print(test_scores)
    return test_scores


system = dict(
    device=-1,
    verbose=5,
    n_jobs=3,
    seed=860
)
data = dict(
    studies=['archi', 'hcp'],
    test_size=0.5,
    train_size=0.5
)
model = dict(
    estimator='logistic',
    normalize=False,
    seed=100,
    target_study=None,
)
factored = dict(
    ensemble=False,
    optimizer='adam',
    latent_size=128,
    activation='linear',
    black_list_target=False,
    regularization=1,
    adaptive_dropout=False,
    sampling='random',
    weight_power=0.6,
    batch_size=128,
    epoch_counting='all',
    init='normal',
    batch_norm=True,
    reset_classifiers=False,
    dropout=0.5,
    input_dropout=0.25,
    l2_penalty=.0,
    seed=100,
    lr={'pretrain': 1e-3, 'train': 1e-3, 'sparsify': 1e-4, 'finetune': 1e-3},
    max_iter={'pretrain': 0, 'train': 200, 'sparsify': 0, 'finetune': 0},
    refit_data=['classifier', 'dropout'])


ensemble = dict(
    n_runs=45,
    n_splits=3,
    alpha=1e-3,
    warmup=False)

logistic = dict(
    estimator='logistic',
    l2_penalty=[7e-5],
    max_iter=1000,
    refit_from=None,)

# Run baseline
# output_dir = join('train/logistic')
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# train(data, system, model, factored, logistic, ensemble)

# Run multi-study model
output_dir = join('train/factored')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model['estimator'] = 'factored'

train(data, system, model, factored, logistic, ensemble)

