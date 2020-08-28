import os
import re
import warnings
from math import ceil
from os.path import join

import pandas as pd
from joblib import load
from sklearn.datasets.base import Bunch

from cogspaces.datasets.utils import get_data_dir

warnings.filterwarnings('ignore', category=FutureWarning, module='h5py')
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir


def fetch_atlas_modl(data_dir=None,
                     url=None,
                     resume=True, verbose=1):
    """Download and load a multi-scale atlas computed using MODL over HCP900.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a non-
        standard location. Default: None (meaning: default)
    url: string, optional
        Download URL of the dataset. Overwrite the default URL.
    """

    if url is None:
        url = 'http://cogspaces.github.io/assets/data/modl/'

    data_dir = get_data_dir(data_dir)
    dataset_name = 'modl'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    keys = ['components_64',
            'components_128',
            'components_453_gm',
            'loadings_128_gm'
            ]

    paths = [
        'components_64.nii.gz',
        'components_128.nii.gz',
        'components_453_gm.nii.gz',
        'loadings_128_gm.npy',
    ]
    urls = [url + path for path in paths]
    files = [(path, url, {}) for path, url in zip(paths, urls)]

    files = _fetch_files(data_dir, files, resume=resume,
                         verbose=verbose)

    params = {key: file for key, file in zip(keys, files)}

    fdescr = 'Components computed using the MODL package, at various scale,' \
             'from HCP900 data'

    params['description'] = fdescr
    params['data_dir'] = data_dir

    return Bunch(**params)


STUDY_LIST = ['knops2009recruitment', 'ds009', 'gauthier2010resonance',
              'ds017B', 'ds110', 'vagharchakian2012temporal', 'ds001',
              'devauchelle2009sentence', 'camcan', 'archi',
              'henson2010faces', 'ds052', 'ds006A', 'ds109', 'ds108', 'la5c',
              'gauthier2009resonance', 'ds011', 'ds107', 'ds116', 'ds101',
              'ds002', 'ds003', 'ds051', 'ds008', 'pinel2009twins', 'ds017A',
              'ds105', 'ds007', 'ds005', 'amalric2012mathematicians', 'ds114',
              'brainomics', 'cauvet2009muslang', 'hcp']


def fetch_reduced_loadings(data_dir=None, url=None, verbose=False,
                           resume=True):
    if url is None:
        url = 'http://cogspaces.github.io/assets/data/loadings/'

    data_dir = get_data_dir(data_dir)
    dataset_name = 'loadings'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    keys = STUDY_LIST

    paths = ['data_%s.pt' % key for key in keys]
    urls = [url + path for path in paths]
    files = [(path, url, {}) for path, url in zip(paths, urls)]

    files = _fetch_files(data_dir, files, resume=resume,
                         verbose=verbose)

    params = {key: file for key, file in zip(keys, files)}

    fdescr = (
        "Z-statistic loadings over a dictionary of 453 components covering "
        "grey-matter `modl_atlas['components_512_gm']` "
        "for 35 different task fMRI studies.")

    params['description'] = fdescr
    params['data_dir'] = data_dir

    return params


def add_study_contrast(ys):
    for study in ys:
        ys[study]['study_contrast'] = ys[study]['study'] + '__' + ys[study]['task'] + '__' + \
                                      ys[study]['contrast']
    return ys

def load_reduced_loadings(data_dir=None, url=None, verbose=False, resume=True):
    loadings = fetch_reduced_loadings(data_dir, url, verbose, resume)
    del loadings['description']
    del loadings['data_dir']
    Xs, ys = {}, {}
    for study, loading in loadings.items():
        Xs[study], ys[study] = load(loading)
    ys = add_study_contrast(ys)
    return Xs, ys


def load_from_directory(dataset, data_dir=None):
    data_dir = get_data_dir(data_dir)
    dataset_dir = join(data_dir, dataset)
    Xs, ys = {}, {}
    regex = re.compile(r'data_(.*).pt')
    for file in os.listdir(dataset_dir):
        m = regex.match(file)
        if m is not None:
            study = m.group(1)
            Xs[study], ys[study] = load(join(dataset_dir, file))
    ys = add_study_contrast(ys)
    return Xs, ys


def fetch_mask(data_dir=None, url=None, resume=True, verbose=1):
    if url is None:
        url = 'http://cogspaces.github.io/assets/data/hcp_mask.nii.gz'
    files = [('hcp_mask.nii.gz', url, {})]

    dataset_name = 'mask'
    data_dir = get_data_dir(data_dir)
    dataset_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                   verbose=verbose)
    files = _fetch_files(dataset_dir, files, resume=resume,
                         verbose=verbose)
    return files[0]


def get_chance_subjects(data_dir=None):
    data, target = load_reduced_loadings(data_dir)
    chance_level = {}
    n_subjects = {}
    for study, this_target in target.items():
        chance_level[study] = 1. / len(this_target['contrast'].unique())
        n_subjects[study] = int(ceil(len(this_target['subject'].unique()) / 2))

    chance_level = pd.Series(chance_level)
    n_subjects = pd.Series(n_subjects)
    return chance_level, n_subjects


def get_brainpedia_descr():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    df = pd.read_csv(join(dirname, 'brainpedia.csv'), index_col=0,
                     header=0)
    return df
