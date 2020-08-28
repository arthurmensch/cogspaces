"""Runs only on dragos"""
from cogspaces.raw_datasets.contrast import fetch_all


def test_fetch_all():
    df = fetch_all()
    print(df)
