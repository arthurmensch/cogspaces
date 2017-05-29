import os


def get_output_dir(data_dir=None):
    """ Returns the directories in which cogspaces store results.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
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
    2. the global environment variable MODL_SHARED_DATA
    3. the user environment variable MODL_DATA
    4. modl_data in the user home folder
    """

    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend(data_dir.split(os.pathsep))

    # If data_dir has not been specified, then we crawl default locations
    if data_dir is None:

        output_dir = os.getenv('OUTPUT_DIR')
        if output_dir is not None:
            paths.extend(output_dir.split(os.pathsep))

        paths.append(os.path.expanduser('~/output'))
    return paths