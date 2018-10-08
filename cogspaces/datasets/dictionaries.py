import os

from nilearn.datasets.utils import _fetch_files, _uncompress_file
from sklearn.datasets.base import Bunch

from .utils import _get_dataset_dir, get_data_dir


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
        url = 'https://team.inria.fr/parietal/files/2018/10/modl_components.zip'

    data_dir = get_data_dir(data_dir)
    dataset_name = 'modl'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = [('modl_components.zip', url, {})]

    keys = ['components_64',
            'components_128',
            'components_512',
            'components_512_gm',
            ]

    paths = [os.path.join(data_dir, key + '.nii.gz') for key in keys]

    if not all(os.path.exists(path) for path in paths):
        zip_file = _fetch_files(data_dir, files, resume=resume,
                                verbose=verbose)[0]
        _uncompress_file(zip_file)

    fdescr = 'Components computed using the MODL package, at various scale,' \
             'from HCP900 data'

    params = dict(zip(keys, paths))
    params['description'] = fdescr
    params['data_dir'] = data_dir

    return Bunch(**params)