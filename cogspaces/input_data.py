import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

idx = pd.IndexSlice


def infinite_iter(iterable):
    """
    Create cycling iterable for finite iterable.

    Parameters
    ----------
    iterable: iterable,
        Input iterable

    Returns
    -------
    iterable:
        Iterable that recreates itself
    """
    while True:
        for elem in iterable:
            yield elem


class RandomChoiceIter:
    """
    Simple iterable that randomly chooses from a list, with probabilitie.

    Parameters
    ----------
    choices :  List,
        List of elements to choose from

    p : List[float]
        Probabilities weights. Must sum to one

    seed : int or None
        Seed the sampler
    """
    def __init__(self, choices, p, seed=None):
        self.random_state = check_random_state(seed)
        self.choices = choices
        self.p = p

    def __next__(self):
        return self.random_state.choice(self.choices, p=self.p)


class MultiStudyLoaderIter:
    """Pytorch loader iterable for a collection of study data.
    """
    def __init__(self, loader):
        data = loader.data
        loaders = {study: DataLoader(this_data,
                                     shuffle=True,
                                     batch_size=loader.batch_size,
                                     pin_memory=loader.device.type == 'cuda')
                   for study, this_data in data.items()}
        self.loader_iters = {study: infinite_iter(loader)
                             for study, loader in loaders.items()}

        studies = list(data.keys())
        self.sampling = loader.sampling
        if self.sampling == 'random':
            p = np.array([loader.study_weights[study] for study in studies])
            assert (np.all(p >= 0))
            p /= np.sum(p)
            self.study_iter = RandomChoiceIter(studies, p, loader.seed)
        elif self.sampling == 'cycle':
            self.study_iter = itertools.cycle(studies)
        elif self.sampling == 'all':
            self.studies = studies
        else:
            raise ValueError('Wrong value for `sampling`')

        self.device = loader.device

    def __next__(self):
        inputs, targets = {}, {}
        if self.sampling == 'all':
            for study in self.studies:
                input, target = next(self.loader_iters[study])
                input = input.to(device=self.device)
                target = target.to(device=self.device)
                inputs[study], targets[study] = input, target
        else:
            study = next(self.study_iter)
            input, target = next(self.loader_iters[study])
            input = input.to(device=self.device)
            target = target.to(device=self.device)
            inputs[study], targets[study] = input, target
        return inputs, targets


class MultiStudyLoader:
    """Pytorch loader for a collection of study data.

    Parameters
    ----------
    data : Dict[str, TensorDataset]
        Collections of TensorDataset, one for each study

    batch_size : int
        Batch size for samples

    sampling : str in {'random', 'cycle', 'all'}
        Sampling strategy.

    study_weights : Dict[str, float]
        Used to sample studies

    seed : int or None
        Seed the sampling of studies

    device : torch.device
        Device to load the data on
    """
    def __init__(self, data,
                 batch_size=128, sampling='cycle',
                 study_weights=None, seed=None, device=torch.device('cpu')):
        self.data = data
        self.batch_size = batch_size
        self.sampling = sampling
        self.study_weights = study_weights
        self.device = device
        self.seed = seed

    def __iter__(self):
        """
        Returns
        -------
        iterable: MultiStudyLoaderIter
            Iterator that yields samples.
        """
        return MultiStudyLoaderIter(self)
