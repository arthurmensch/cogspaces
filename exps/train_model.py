# Load data
import json
import os
from os.path import join

from joblib import dump, Memory
from sklearn.metrics import accuracy_score

from cogspaces.classification.factored import FactoredClassifier
from cogspaces.classification.factored_dl import FactoredDL
from cogspaces.classification.logistic import MultiLogisticClassifier
from cogspaces.datasets import STUDY_LIST, load_reduced_loadings
from cogspaces.datasets.contrast import load_masked_contrasts
from cogspaces.model_selection import train_test_split
from cogspaces.preprocessing import MultiStandardScaler, MultiTargetEncoder
from cogspaces.utils import compute_metrics, ScoreCallback, MultiCallback

# Parameters
system = dict(
    verbose=5,
    n_jobs=3,
    seed=860
)
data = dict(
    studies=['archi', 'hcp'],
    test_size=0.5,
    train_size=0.5,
    reduced=True,
    data_dir=None,
)
model = dict(
    estimator='logistic',
    normalize=False,
    seed=100,
    target_study=None,
)

if model['estimator'] in ['factored', 'ensemble']:
    factored = dict(
        latent_size=128,
        weight_power=0.6,
        batch_size=128,
        init='normal',
        dropout=0.5,
        input_dropout=0.25,
        seed=100,
        lr={'pretrain': 1e-3, 'train': 1e-3, 'finetune': 1e-3},
        max_iter={'pretrain': 0, 'train': 200, 'finetune': 0},
        )
    if model['estimator'] == 'ensemble':
        ensemble = dict(
            n_runs=45,
            n_splits=3,
            alpha=1e-3,
            warmup=False)
else:
    logistic = dict(
        estimator='logistic',
        l2_penalty=[7e-5],
        max_iter=1000,
        refit_from=None,)

output_dir = join('output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
info = {}

# Load data
if data['studies'] == 'all':
    studies = STUDY_LIST
elif isinstance(data['studies'], str):
    studies = [data['studies']]
elif isinstance(data['studies'], list):
    studies = data['studies']
else:
    raise ValueError("Studies should be a list or 'all'")

if data['reduced']:
    input_data, target = load_reduced_loadings(data_dir=data['data_dir'])
else:
    input_data, target = load_masked_contrasts(data_dir=data['data_dir'])

input_data = {study: input_data[study] for study in studies}
target = {study: target[study] for study in studies}

target_encoder = MultiTargetEncoder().fit(target)
target = target_encoder.transform(target)

train_data, test_data, train_targets, test_targets = \
    train_test_split(input_data, target, random_state=system['seed'],
                     test_size=data['test_size'],
                     train_size=data['train_size'])

if model['normalize']:
    standard_scaler = MultiStandardScaler().fit(train_data)
    train_data = standard_scaler.transform(train_data)
    test_data = standard_scaler.transform(test_data)
else:
    standard_scaler = None


if model['estimator'] in ['factored', 'ensemble']:
    estimator = FactoredClassifier(verbose=system['verbose'],
                                   n_jobs=system['n_jobs'],
                                   **factored)
    if model == 'ensemble':
        memory = Memory(cachedir='cache')
        estimator = FactoredDL(estimator,
                               n_jobs=system['n_jobs'],
                               n_runs=ensemble['n_runs'],
                               alpha=ensemble['alpha'],
                               warmup=ensemble['warmup'],
                               seed=factored['seed'],
                               memory=memory,
                               )
    else:
        # Set some callback to obtain useful verbosity
        test_callback = ScoreCallback(Xs=test_data, ys=test_targets,
                                      score_function=accuracy_score)
        train_callback = ScoreCallback(Xs=train_data, ys=train_targets,
                                       score_function=accuracy_score)
        callback = MultiCallback({'train': train_callback,
                                  'test': test_callback})
        info['n_iter'] = train_callback.n_iter_
        info['train_scores'] = train_callback.scores_
        info['test_scores'] = test_callback.scores_
elif model['estimator'] == 'logistic':
    estimator = MultiLogisticClassifier(verbose=system['verbose'],
                                        **logistic)
    callback = None
else:
    raise ValueError("Wrong estimator parameter")


estimator.fit(train_data, train_targets, callback=callback)

test_preds = estimator.predict(test_data)

scores = compute_metrics(test_preds, test_targets, target_encoder)

dump(target_encoder, join(output_dir, 'target_encoder.pkl'))
dump(standard_scaler, join(output_dir, 'standard_scaler.pkl'))
dump(estimator, join(output_dir, 'estimator.pkl'))
with open(join(output_dir, 'metrics.json'), 'w+') as f:
    json.dump(info, f)
with open(join(output_dir, 'info.json'), 'w+') as f:
    json.dump(info, f)