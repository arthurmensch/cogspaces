import os
from os.path import join, expanduser

from joblib import Memory
from sklearn.metrics import accuracy_score

from cogspaces.classification.factored import FactoredClassifier
from cogspaces.classification.factored_dl import FactoredDL
from cogspaces.datasets import STUDY_LIST, load_reduced_loadings
from cogspaces.datasets.contrast import load_masked_contrasts
from cogspaces.model_selection import train_test_split
from cogspaces.preprocessing import MultiStandardScaler, MultiTargetEncoder
from cogspaces.report import save
from cogspaces.utils import compute_metrics, ScoreCallback, MultiCallback

output_dir = expanduser(join('~', 'output', 'factored'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameters
system = dict(
    verbose=1,
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
    estimator='factored',
    normalize=False,
    seed=100,
    target_study=None,
)


factored = dict(
    latent_size=128,
    weight_power=0.6,
    batch_size=128,
    init='normal',
    dropout=0.5,
    input_dropout=0.25,
    seed=100,
    lr={'pretrain': 1e-3, 'train': 1e-3, 'finetune': 1e-3},
    max_iter={'pretrain': 0, 'train': 2, 'finetune': 0},
    )

config = {'system': system, 'data': data, 'model': model, 'factored': factored}

if model['estimator'] == 'ensemble':
    ensemble = dict(
        n_runs=45,
        n_splits=3,
        alpha=1e-3,
        warmup=False)
    config['ensemble'] = ensemble
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


# Train
if model['normalize']:
    standard_scaler = MultiStandardScaler().fit(train_data)
    train_data = standard_scaler.transform(train_data)
    test_data = standard_scaler.transform(test_data)
else:
    standard_scaler = None

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


estimator.fit(train_data, train_targets, callback=callback)

# Estimate
test_preds = estimator.predict(test_data)
metrics = compute_metrics(test_preds, test_targets, target_encoder)

# Save model for further analysis
save(target_encoder, standard_scaler, estimator, metrics, info, config, output_dir)