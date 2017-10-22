import pandas as pd

import os
from os.path import join

from sklearn.externals.joblib import load

from cogspaces.pipeline import get_output_dir

unmasked_dir = join(get_output_dir(), 'reduced')
reduced_dir = join(get_output_dir(), 'unmasked')

for base_dir in [unmasked_dir, reduced_dir]:
    for root, dirs, files in os.walk(reduced_dir):
        for this_file in files:
            if this_file in ['Xt.pkl', 'imgs.pkl']:
                this_file = join(root, this_file)
                print(this_file)
                Xt = load(this_file)
                Xt.to_pickle(this_file)

