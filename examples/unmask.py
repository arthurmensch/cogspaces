from os.path import expanduser, join

from joblib import Memory, dump

from cogspaces.datasets import fetch_archi, fetch_mask, fetch_atlas_modl
from cogspaces.datasets.studies import fetch_contrasts, fetch_all
from cogspaces.input_data import make_array_data
from cogspaces.pipeline import get_output_dir

mem = Memory(cachedir=expanduser('~/cache'))

mask = fetch_mask()
maps = fetch_atlas_modl().components128

dataframe = fetch_all()
X = make_array_data(dataframe['z_map'], mask=mask, n_jobs=5)
y = dataframe.index.values
dump((X, y), join(get_output_dir(), 'unmasked_data.pkl'))
