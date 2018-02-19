# Load data
from os.path import join

import torch
from cogspaces.datasets.utils import get_data_dir
from cogspaces.models.multi_layer import MultiClassifier
from cogspaces.utils.data import load_prepared_data
from torch.autograd import Variable
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader


def repeat(data_loader):
    while True:
        for elem in data_loader:
            yield elem


def accuracy_score(pred, target):
    _, pred = pred.max(dim=1)
    return torch.mean((pred == target).float()).data[0]


max_iter = 10000

prepared_data_dir = join(get_data_dir(), 'reduced_512')
datasets, target_encoder, n_features, target_sizes \
    = load_prepared_data(data_dir=prepared_data_dir, torch=True)

datasets = {study: datasets[study] for study in ['hcp', 'archi']}

train_data_loaders = {study: repeat(DataLoader(dataset['train'],
                                               shuffle=True,
                                               batch_size=64)) for
                      study, dataset in datasets.items()}
test_data_loaders = {study: repeat(
    DataLoader(dataset['test'], shuffle=False,
               batch_size=len(dataset['test'])))
                     for study, dataset in datasets.items()}

model = MultiClassifier(in_features=n_features, target_sizes=target_sizes,
                        embedding_size=50, input_dropout=0.25,
                        dropout=.5)
loss_function = NLLLoss()

optimizer = Adam(model.parameters())

n_iter = 0
while n_iter < max_iter:
    model.zero_grad()
    loss = 0
    accuracies = {}
    test_accuracies = {}
    for study, loader in train_data_loaders.items():
        data, (studies, subjects, contrasts) = next(loader)
        data = Variable(data)
        contrasts = Variable(contrasts.squeeze())
        preds = model({study: data})[study]
        loss += loss_function(preds, contrasts)
        accuracies[study] = accuracy_score(preds, contrasts)

    for study, loader in train_data_loaders.items():
        data, (studies, subjects, contrasts) = next(loader)
        data = Variable(data)
        contrasts = Variable(contrasts.squeeze())
        preds = model({study: data})[study]
        test_accuracies[study] = accuracy_score(preds, contrasts)


    loss.backward()
    optimizer.step()
    n_iter += 1
    print('Iteration %s, train loss %f' % (n_iter, loss.data[0]))
    print('Train accuracy:')
    for study, accuracy in accuracies.items():
        print('%s: %.4f' % (study, accuracy))
    print('Test accuracy:')
    for study, accuracy in test_accuracies.items():
        print('%s: %.4f' % (study, accuracy))