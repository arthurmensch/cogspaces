# Load data
import warnings
from os.path import join

from cogspaces.datasets.utils import get_data_dir
from cogspaces.models.multi_layer import MultiClassifier
from cogspaces.utils.data import load_data, train_test_split, MultiTargetEncoder, \
    MultiStandardScaler
from joblib import dump, load

warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module=r'.*.label')

prepared_data_dir = join(get_data_dir(), 'reduced_512')
data, target = load_data(data_dir=prepared_data_dir)
data = {study: data[study] for study in ['hcp', 'archi']}
target = {study: target[study] for study in ['hcp', 'archi']}


target_encoder = MultiTargetEncoder().fit(target)
target = target_encoder.transform(target)

train_data, test_data, train_targets, test_targets = train_test_split(data,
                                                                      target)
data_transformer = MultiStandardScaler().fit(train_data)
train_data = data_transformer.transform(train_data)
test_data = data_transformer.transform(test_data)

train_contrasts = {study: train_target['contrast'] for study, train_target
                   in train_targets.items()}
test_contrasts = {study: test_target['contrast'] for study, test_target
                  in test_targets.items()}

estimator = MultiClassifier(optimizer='adam', embedding_size=100,
                            batch_size=128,
                            dropout=0.75, max_iter=100000,
                            report_every=1000)
estimator.fit(train_data, train_contrasts, X_val=test_data,
              y_val=test_contrasts)
test_pred_contrasts = estimator.predict(test_data)
test_scores = estimator.score(test_data, test_contrasts)

test_preds = {}
for study, test_target in test_targets.items():
    test_preds[study] = test_target.copy()
    test_preds[study]['contrast'] = test_pred_contrasts[study]


test_preds = target_encoder.inverse_transform(test_preds)
dump(estimator, 'estimator.pkl')
dump(test_preds, 'preds.pkl')
