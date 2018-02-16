import torch
from nilearn.input_data import NiftiMasker
from torch.autograd import Variable
from torch.nn import NLLLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from cogspaces.datasets.contrasts import fetch_archi
from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask
from cogspaces.models.multi_layer import MultiLayerClassifier
from cogspaces.utils.data import tfMRIDataset


def repeat(data_loader):
    while True:
        for elem in data_loader:
            yield elem


def get_first_layer_weights():
    modl_atlas = fetch_atlas_modl()
    mask = fetch_mask()
    dictionary = modl_atlas.components512
    masker = NiftiMasker(mask_img=mask).fit()
    weights = masker.transform(dictionary)
    return torch.from_numpy(weights).float()


# data = fetch_archi(data_dir=)

dataset = tfMRIDataset(data)
dataset.set_target_encoder()
target_sizes = dataset.target_sizes()
n_features = dataset.n_features()

train_dataset, test_dataset = dataset.train_test_split()

train_data_iters = {study: repeat(DataLoader(dataset, shuffle=True,
                                             batch_size=32))
                    for study, dataset in train_dataset.items()}
test_dataloaders = {study: DataLoader(dataset, shuffle=True,
                                      batch_size=128) for study, dataset in
                    test_dataset.items()}

model = MultiLayerClassifier(n_features, target_sizes,
                             first_hidden_features=512,
                             second_hidden_features=None,
                             dropout=0.)


model.first_linear.weight.data = get_first_layer_weights()
model.first_linear.weight.requires_grad = False

loss_function = NLLLoss()
optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3)

current_batch = 0
# Train loop
while current_batch < 100:
    model.zero_grad()
    loss = 0
    data_dict = {}
    contrasts_dict = {}
    for study, data_iter in train_data_iters.items():
        studies, subjects, contrasts, data = next(data_iter)
        data_dict[study] = Variable(data)
        contrasts_dict[study] = Variable(contrasts)
    preds = model(data_dict)
    for study in contrasts_dict:
        loss += loss_function(preds[study], contrasts_dict[study])
    loss.backward()
    optimizer.step()
    current_batch += 1
    print('Iter %s, loss %f' % (current_batch, loss.data[0]))
