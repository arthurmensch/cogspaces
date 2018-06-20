import json
import numpy as np
import os
import re
from joblib import load
from os.path import join

from cogspaces.datasets.utils import get_output_dir
from cogspaces.utils import get_dictionary, get_masker

output_dir = join(get_output_dir(), 'logistic_gm_full')

regex = re.compile(r'[0-9]+$')
res = []
names = []
coefs = []

for this_dir in filter(regex.match, os.listdir(output_dir)):
    this_exp_dir = join(output_dir, this_dir)
    this_dir = int(this_dir)
    config = json.load(
        open(join(this_exp_dir, 'config.json'), 'r'))
    run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
    study = config['data']['studies']
    target_encoder = load(join(this_exp_dir, 'target_encoder.pkl'), 'r')
    estimator = load(join(this_exp_dir, 'estimator.pkl'), 'r')
    contrasts = target_encoder.le_[study]['contrast'].classes_
    coefs.append(estimator.coefs_[study])
    names.append(['%s::%s' % (study, contrast) for contrast in contrasts])
coefs = np.concatenate(coefs, axis=0)
dictionary = get_dictionary()
masker = get_masker()
components = coefs.dot(dictionary)
img = masker.inverse_transform(components)

img.to_filename(join(output_dir, 'components.nii.gz'))