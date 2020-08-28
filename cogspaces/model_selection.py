import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from cogspaces.utils import zip_data, unzip_data


def train_test_split(data, target, test_size=.5, train_size=.5,
                     random_state=0):
    """
        Takes a collection of datasets and split them into a test and train
        collection.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Collection of input data (one for each study)

    target : Dict[str, pd.DataFrame]
        Collection of targets (one for each study)

    test_size : float in [0, 1]  or Dict[str, float]
        Test size for each study

    train_size : float in [0, 1] or Dict[str, float]
        Train size for each study

    random_state: int or None
        Seed for the split

    Returns
    -------
    train_data : Dict[str, np.ndarray]

    test_data : Dict[str, np.ndarray]

    train_target : Dict[str, pd.DataFrame]

    test_target : Dict[str, pd.DataFrame]
    """
    data = zip_data(data, target)
    datasets = {'train': {}, 'test': {}}

    if isinstance(test_size, (float, int)):
        test_size = {study: test_size for study in data}
    if isinstance(train_size, (float, int)):
        train_size = {study: train_size for study in data}

    splits = {}
    for study, (this_data, this_target) in data.items():
        if train_size[study] == 1.:
            train = np.arange(len(this_data))
            test = [0]
        else:
            cv = GroupShuffleSplit(n_splits=1, test_size=test_size[study],
                                   train_size=train_size[study],
                                   random_state=random_state)
            train, test = next(cv.split(X=this_data,
                                        groups=this_target['subject']))
        for fold, indices in [('train', train), ('test', test)]:
            datasets[fold][study] = this_data[indices], \
                                    this_target.iloc[indices]
    train_data, train_target = unzip_data(datasets['train'])
    test_data, test_target = unzip_data(datasets['test'])
    return train_data, test_data, train_target, test_target