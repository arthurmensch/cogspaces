import os
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir


def get_data_dir(data_dir=None):
    """ Returns the directories in which to look for utils.

    This is typically useful for the end-user to check where the utils is
    downloaded and stored.

    Parameters
    ----------
    data_dir: string, optional
        Path of the utils directory. Used to force utils storage in a specified
        location. Default: None

    Returns
    -------
    path: string
        Path of the dataset directories.

    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :
    1. the keyword argument data_dir
    4. /storage/store/data
    """

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        assert (isinstance(data_dir, str))
        return data_dir
    elif 'COGSPACES_DATA' in os.environ:
        return os.environ['COGSPACES_DATA']
    else:
        return '/storage/store/data/cogspaces'


def get_output_dir(data_dir=None):
    """ Returns the directories in which cogspaces store results.

    Parameters
    ----------
    data_dir: string, optional
        Path of the utils directory. Used to force utils storage in a specified
        location. Default: None

    Returns
    -------
    paths: list of strings
        Paths of the dataset directories.

    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :
    1. the keyword argument data_dir
    2. the global environment variable OUTPUT_COGSPACES_DIR
    4. output/cogspaces in the user home folder
    """

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        return data_dir
    else:
        # If data_dir has not been specified, then we crawl default locations
        output_dir = os.getenv('OUTPUT_COGSPACES_DIR')
        if output_dir is not None:
            return output_dir
    return os.path.expanduser('~/output/cogspaces')


def fetch_mask(data_dir=None, url=None, resume=True, verbose=1):
    if url is None:
        url = 'http://www.amensch.fr/data/cogspaces/mask/'

    files = ['mask_img.nii.gz']

    if isinstance(url, str):
        url = [url] * len(files)

    files = [(f, u + f, {}) for f, u in zip(files, url)]

    dataset_name = 'mask'
    data_dir = get_data_dir(data_dir)
    dataset_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                   verbose=verbose)
    files_ = _fetch_files(dataset_dir, files, resume=resume,
                          verbose=verbose)
    return files_[0]
