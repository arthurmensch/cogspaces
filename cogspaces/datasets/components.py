from nilearn.datasets.utils import _fetch_files
from sklearn.datasets.base import Bunch
from .utils import _get_dataset_dir


def fetch_atlas_modl(data_dir=None, url=None,
                     resume=True, verbose=1):
    """Download and load a multi-scale atlas computed using MODL over HCP900.

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

        - 20-dimensional ICA, Resting-FMRI components:

          - all 20 components (rsn20)
          - 10 well-matched maps from these, as shown in PNAS paper (rsn10)

        - 20-dimensional ICA, BrainMap components:

          - all 20 components (bm20)
          - 10 well-matched maps from these, as shown in PNAS paper (bm10)

        - 70-dimensional ICA, Resting-FMRI components (rsn70)

        - 70-dimensional ICA, BrainMap components (bm70)


    References
    ----------

    S.M. Smith, P.T. Fox, K.L. Miller, D.C. Glahn, P.M. Fox, C.E. Mackay, N.
    Filippini, K.E. Watkins, R. Toro, A.R. Laird, and C.F. Beckmann.
    Correspondence of the brain's functional architecture during activation and
    rest. Proc Natl Acad Sci USA (PNAS), 106(31):13040-13045, 2009.

    A.R. Laird, P.M. Fox, S.B. Eickhoff, J.A. Turner, K.L. Ray, D.R. McKay, D.C
    Glahn, C.F. Beckmann, S.M. Smith, and P.T. Fox. Behavioral interpretations
    of intrinsic connectivity networks. Journal of Cognitive Neuroscience, 2011

    Notes
    -----
    For more information about this dataset's structure:
    http://www.fmrib.ox.ac.uk/datasets/brainmap+rsns/
    """
    if url is None:
        url = 'http://www.amensch.fr/data/cogspaces/modl/'

    files = [
        'components_16.nii.gz',
        'components_64.nii.gz',
        'components_128.nii.gz',
        'components_512.nii.gz',
    ]

    if isinstance(url, str):
        url = [url] * len(files)

    files = [(f, u + f, {}) for f, u in zip(files, url)]

    dataset_name = 'modl'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files_ = _fetch_files(data_dir, files, resume=resume,
                          verbose=verbose)

    fdescr = 'Components computed using the MODL package, at various scale,' \
             'from HCP900 data'

    keys = ['components16',
            'components64',
            'components128',
            'components512',
            ]

    params = dict(zip(keys, files_))
    params['description'] = fdescr

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