import numpy as np
from joblib import dump
from os.path import join

from cogspaces.data import load_data_from_dir
from cogspaces.datasets.utils import get_data_dir
from cogspaces.utils import get_dictionary

data, target = load_data_from_dir(data_dir=join(get_data_dir(), 'masked'))

output_dir = join(get_data_dir(), 'masked_gm')
# os.makedirs(output_dir)
dictionary = get_dictionary()
mask = np.any(dictionary, axis=0)
res = []

for study in data:
    this_data = data[study][:, mask]
    this_target = target[study]
    dump((this_data, this_target), join(output_dir, 'data_%s.pt' % study))
