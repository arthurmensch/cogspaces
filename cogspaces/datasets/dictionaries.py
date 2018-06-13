from nilearn.datasets.utils import _fetch_files
from sklearn.datasets.base import Bunch

from .utils import _get_dataset_dir, get_data_dir


def fetch_atlas_modl(data_dir=None, url=None,
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
        url = 'http://amensch.fr/data/cogspaces/modl/'

    files = [
        'components_16.nii.gz',
        'components_128.nii.gz',
        'components_128_small.nii.gz',
        'components_208.nii.gz',
        'components_512.nii.gz',
        'components_512_gm.nii.gz',
        'loadings_128.npy',
        'loadings_128_small.npy',
        'loadings_128_gm.npy',
        'assign_512.npy'
    ]

    if isinstance(url, str):
        url = [url] * len(files)

    files = [(f, u + f, {}) for f, u in zip(files, url)]

    data_dir = get_data_dir()
    dataset_name = 'modl'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files_ = _fetch_files(data_dir, files, resume=resume,
                          verbose=verbose)

    fdescr = 'Components computed using the MODL package, at various scale,' \
             'from HCP900 data'

    keys = ['components16',
            'components128',
            'components128_small',
            'components208',
            'components512',
            'components512_gm',
            'loadings128',
            'loadings128_small',
            'loadings128_gm',
            'assign512'
            ]

    params = dict(zip(keys, files_))
    params['description'] = fdescr
    params['data_dir'] = data_dir

    return Bunch(**params)


def fetch_craddock_parcellation(data_dir=None, url=None,
                                resume=True, verbose=1):
    """Download and load the craddock parcellation.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a non-
        standard location. Default: None (meaning: default)
    mirror: string, optional
        By default, the dataset is downloaded from the original website of the
        atlas. Specifying "nitrc" will force download from a mirror, with
        potentially higher bandwith.
    url: string, optional
        Download URL of the dataset. Overwrite the default URL.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, contains:

        - 200-components parcelellation  (parcellate200)
        - 400-components parcelellation (parcellate400)


    References
    ----------

    ?
    """
    if url is None:
        url = 'http://www.amensch.fr/data/craddock_parcellation/'

    files = [
        'ADHD200_parcellate_200.nii.gz',
        'ADHD200_parcellate_400.nii.gz',
    ]

    if isinstance(url, str):
        url = [url] * len(files)

    files = [(f, u + f, {}) for f, u in zip(files, url)]

    dataset_name = 'craddock_parcellation'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files_ = _fetch_files(data_dir, files, resume=resume,
                          verbose=verbose)

    fdescr = 'Components from Craddock clustering atlas'

    keys = ['parcellate200', 'parcellate400']
    params = dict(zip(keys, files_))
    params['description'] = fdescr

    return Bunch(**params)