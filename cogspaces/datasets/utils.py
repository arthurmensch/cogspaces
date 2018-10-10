import os


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
        return os.path.expanduser('~/cogspaces_data')