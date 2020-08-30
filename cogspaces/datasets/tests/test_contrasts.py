from cogspaces.datasets.contrasts import fetch_all


def test_fetch_all():
    df = fetch_all()
    print(df)