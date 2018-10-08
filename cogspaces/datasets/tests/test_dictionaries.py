import os

from cogspaces.datasets.dictionaries import fetch_atlas_modl


def test_atlas_modl():
    """Download the dataset in default directory and check success.

    """
    atlas = fetch_atlas_modl()
    keys = ['components_64',
            'components_128',
            'components_512',
            'components_512_gm',
            ]

    assert all(os.path.exists(atlas[key]) for key in keys)