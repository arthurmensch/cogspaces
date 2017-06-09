from os.path import join
import numpy as np
from cogspaces.utils import get_output_dir, make_data_frame, split_folds, \
    MultiDatasetTransformer


def test_make_data_frame():
    df = make_data_frame(['archi', 'hcp'], 'hcp_rs_positive',
                         reduced_dir=join(get_output_dir(), 'reduced'))
    datasets = df.index.get_level_values('dataset').unique().values
    assert('archi' in datasets and 'hcp' in datasets and len(datasets) == 2)


def test_split_folds():
    df = make_data_frame(['archi', 'hcp'], 'hcp_rs_positive',
                         reduced_dir=join(get_output_dir(), 'reduced'),
                         n_subjects=10)
    df_train, df_test = split_folds(df, test_size=0.2, train_size=None,
                                    random_state=1)
    n_train = df_train.shape[0]
    n_test = df_test.shape[0]
    n_samples = df.shape[0]
    assert(n_test == 0.2 * n_samples)
    assert(n_train == 0.8 * n_samples)


def test_multi_dataset_transformer():
    transformer = MultiDatasetTransformer()
    df = make_data_frame(['archi', 'hcp'], 'hcp_rs_positive',
                         reduced_dir=join(get_output_dir(), 'reduced'),
                         n_subjects=10)
    X, y = transformer.fit_transform(df)
    assert(len(X) == 2 and len(y) == 2)
    contrasts = transformer.inverse_transform(df, y).values
    true_contrasts = df.index.get_level_values('contrast').values
    assert(np.all(contrasts == true_contrasts))


def test_pipeline():
    df = make_data_frame(['archi', 'hcp'], 'hcp_rs_positive',
                         reduced_dir=join(get_output_dir(), 'reduced'),
                         n_subjects=10)
    df_train, df_test = split_folds(df, test_size=0.2, train_size=None,
                                    random_state=1)
    transformer = MultiDatasetTransformer()
    transformer.fit_transform(df_train)
    X_test, y_test = transformer.fit_transform(df_test)
    contrasts = transformer.inverse_transform(df_test, y_test).values
    true_contrasts = df_test.index.get_level_values('contrast').values
    assert(np.all(contrasts == true_contrasts))