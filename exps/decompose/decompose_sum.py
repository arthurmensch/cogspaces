import pandas as pd

import json
import os
from json import JSONDecodeError
from os.path import join

import numpy as np
from nilearn.input_data import NiftiMasker

from cogspaces.datasets import fetch_mask
from cogspaces.pipeline import get_output_dir

basedir = join(get_output_dir(), 'multi_decompose', '3', 'run')
mask = fetch_mask()
masker = NiftiMasker(mask_img=mask).fit()


res = []
for exp_dir in os.listdir(basedir):
    print(basedir)
    try:
        id_exp = int(exp_dir)
    except:
        continue
    exp_dir = join(basedir, exp_dir)
    artifact_dir = join(get_output_dir(), 'decompose', str(id_exp), 'artifacts')
    try:
        config = json.load(open(join(exp_dir, 'config.json'), 'r'))
        info = json.load(open(join(exp_dir, 'info.json'), 'r'))
    except (JSONDecodeError, FileNotFoundError):
        continue
    n_components = config['n_components']
    alpha = config['alpha']
    img = join(artifact_dir, 'components.nii.gz')
    try:
        data = masker.transform(img)
    except:
        continue
    n_covering_pixels = data.sum(axis=0)
    non_covered_pixels = np.sum(n_covering_pixels == 0)
    res.append(dict(non_covered_pixels=non_covered_pixels, alpha=alpha,
                    n_components=n_components, img=img, id_exp=id_exp))
res = pd.DataFrame(res)
print(res)
res.set_index(['id_exp'])
res.sort_index(inplace=True)
res.to_pickle(join(get_output_dir(), 'multi_decompose',
                   '3', 'decompose_sum.pkl'))

df = pd.read_pickle(join(get_output_dir(), 'multi_decompose',
                    '3', 'decompose_sum.pkl'))
df = df.sort_values(by=['n_components', 'alpha'])
print(df)