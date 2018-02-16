from os.path import join

from cogspaces.datasets.utils import get_data_dir
from cogspaces.models.multi_layer import MultiLayerClassifier
from cogspaces.utils.data import load_dataset
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader


def infinite_data_loader(data_loader):
    while True:
        for data in data_loader:
            yield data


datasets = load_dataset(join(get_data_dir(), 'prepared_seed_0'))
loaders = {'train': {}, 'test': {}}
for fold in datasets:
    for dataset in datasets[fold]:
        if fold == 'train':
            loaders[fold][dataset] = infinite_data_loader(
                DataLoader(datasets[fold][dataset],
                           shuffle=True, batch_size=32, num_workers=4))
        else:
            loaders[fold][dataset] = DataLoader(datasets[fold][dataset],
                                                shuffle=False,
                                                batch_size=32)
n_features = next(iter(datasets['train'].values())).n_features()
target_sizes = {name: dataset.n_contrasts()
                for name, dataset in datasets['train'].items()}
print(target_sizes)
model = MultiLayerClassifier(n_features, target_sizes,
                             first_hidden_features=512,
                             second_hidden_features=10,
                             dropout=0.5)
loss_function = NLLLoss()
optimizer = Adam()

# Train loop
train_loaders = loaders['train']
n_iter = 0
while n_iter < 100:
    Xs = {}
    ys = {}
    for dataset in train_loaders:
        loader = train_loaders[dataset]
        X, y = next(loader)
        Xs[dataset] = X
        ys[dataset] = y
    preds = model(Xs)
    loss = 0
    for dataset in preds:
        loss += loss_function(preds[dataset], ys[dataset])
    model.zero_grad()
    loss.backward()
    optimizer.step()
    n_iter += 1
    print(n_iter, loss)