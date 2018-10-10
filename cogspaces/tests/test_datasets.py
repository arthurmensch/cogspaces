import os

from cogspaces.datasets import fetch_atlas_modl, fetch_mask, \
    fetch_reduced_loadings, STUDY_LIST, fetch_contrasts


def test_atlas_modl():
    """Download the dataset in default directory and check success.

    """
    atlas = fetch_atlas_modl()
    keys = ['components_64',
            'components_128',
            'components_453_gm',
            ]

    assert all(os.path.exists(atlas[key]) for key in keys)


def test_fetch_mask():
    fetch_mask()


def test_reduced_loadings():
    loadings = fetch_reduced_loadings()
    keys = STUDY_LIST

    assert all(os.path.exists(loadings[key]) for key in keys)


def test_statistics():
    df =  fetch_contrasts('brainpedia')
    print(df)
