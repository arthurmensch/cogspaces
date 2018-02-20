# Load data
import os
import warnings
from os.path import join

from joblib import dump
from sacred import Experiment

from cogspaces.datasets.utils import get_data_dir, get_output_dir
from cogspaces.models.baseline import MultiLogisticClassifier
from cogspaces.metrics import accuracy
from cogspaces.models.factored import FactoredClassifier
from cogspaces.models.trace import TraceClassifier
from cogspaces.data import load_data
from cogspaces.model_selection import train_test_split
from cogspaces.preprocessing import MultiStandardScaler, MultiTargetEncoder

warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module=r'.*.label')

exp = Experiment('multi_studies')


@exp.config
def system_config():
    device = -1
    source_dir = join(get_data_dir(), 'reduced_512')
    output_dir = join(get_output_dir(), 'multi_dataset')
    seed = 0


@exp.config
def config():
    studies = ['archi', 'hcp']
    normalize = True
    model = 'factored'


@exp.automain
def train(source_dir, device, output_dir,
          studies, normalize, model, _seed):

    data, target = load_data(data_dir=source_dir)

    if isinstance(studies, list):
        data = {study: data[study] for study in studies}
        target = {study: target[study] for study in studies}

    target_encoder = MultiTargetEncoder().fit(target)
    target = target_encoder.transform(target)

    train_data, test_data, train_targets, test_targets = \
        train_test_split(data, target, random_state=_seed)

    if normalize:
        standard_scaler = MultiStandardScaler().fit(train_data)
        train_data = standard_scaler.transform(train_data)
        test_data = standard_scaler.transform(test_data)

    train_contrasts = {study: train_target['contrast'] for study, train_target
                       in train_targets.items()}
    test_contrasts = {study: test_target['contrast'] for study, test_target
                      in test_targets.items()}

    if model == 'factored':
        estimator = FactoredClassifier(optimizer='lbfgs',
                                       embedding_size=53,
                                       batch_size=128,
                                       dropout=0.0,
                                       input_dropout=0.0,
                                       l2_penalty=1e-3,
                                       max_iter=10,
                                       report_every=10,
                                       device=device)
    elif model == 'trace':
        estimator = TraceClassifier(trace_penalty=1e-3, verbose=100, max_iter=1000,
                                    step_size_multiplier=100)
    elif model == 'logistic':
        estimator = MultiLogisticClassifier(l2_penalty=100, verbose=100)
    else:
        return ValueError("Wrong value for parameter `model`: got '%s'."
                          % model)

    estimator.fit(train_data, train_contrasts, X_val=test_data,
                  y_val=test_contrasts)
    test_pred_contrasts = estimator.predict(test_data)
    test_scores = accuracy(test_pred_contrasts, test_contrasts)

    test_preds = {}
    for study, test_target in test_targets.items():
        test_preds[study] = test_target.copy()
        test_preds[study]['contrast'] = test_pred_contrasts[study]
    test_preds = target_encoder.inverse_transform(test_preds)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dump(estimator, join(output_dir, 'estimator.pkl'))
    dump(test_preds, join(output_dir, 'preds.pkl'))
    dump(target_encoder, join(output_dir, 'target_encoder.pkl'))
    if normalize:
        dump(standard_scaler, join(output_dir, 'standard_scaler.pkl'))

    return test_scores
