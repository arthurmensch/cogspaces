from sklearn.model_selection import GroupShuffleSplit

from cogspaces.utils import zip_data, unzip_data


def train_test_split(data, target, test_size=.5,
                     random_state=0):
    data = zip_data(data, target)
    datasets = {'train': {}, 'test': {}}
    for study, (this_data, this_target) in data.items():
        cv = GroupShuffleSplit(n_splits=1, test_size=test_size,
                               random_state=random_state)
        train, test = next(cv.split(X=this_data,
                                    groups=this_target['subject']))
        for fold, indices in [('train', train), ('test', test)]:
            datasets[fold][study] = this_data[indices], \
                                    this_target.iloc[indices]
    train_data, train_target = unzip_data(datasets['train'])
    test_data, test_target = unzip_data(datasets['test'])
    return train_data, test_data, train_target, test_target