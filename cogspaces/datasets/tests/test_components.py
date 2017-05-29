from cogspaces.datasets import fetch_craddock_parcellation, fetch_atlas_modl


def test_craddock_parcellation():
    data = fetch_craddock_parcellation()
    print(data)

    data = fetch_atlas_modl()
    print(data)
