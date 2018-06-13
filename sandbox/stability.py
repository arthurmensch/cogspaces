import numpy as np
import os
import pandas as pd
import re
from joblib import Memory, Parallel, delayed
from nilearn.input_data import MultiNiftiMasker
from os.path import expanduser
from os.path import join
from sklearn.utils.linear_assignment_ import linear_assignment

from cogspaces.datasets.utils import fetch_mask

idx = pd.IndexSlice


def relative_stability(comp1, comp2):
    comp1 = comp1 / np.sqrt(np.sum(comp1 ** 2, axis=1, keepdims=True))
    comp2 = comp2 / np.sqrt(np.sum(comp2 ** 2, axis=1, keepdims=True))
    Q = comp2.dot(comp1.T)
    assign = linear_assignment(-np.abs(Q))[:, 1]
    Q = Q[:, assign]
    return np.abs(np.diag(Q)).tolist()


def compute_stabilities(components):
    res = Parallel(n_jobs=10, verbose=10)(delayed(relative_stability)(
        this_comp, other_comp)
                        for this_comp in components for other_comp in components)

    all_stability = []
    n = len(components)
    for this_comp in components:
        stability = []
        for other_comp in components:
            if other_comp is not this_comp:
                stability.append(res[0])
            res = res[1:]
        all_stability.append(stability)
    all_stability = np.array(all_stability)
    return all_stability

mem = Memory(cachedir=expanduser('~/cache_local'))
output_dir = expanduser('~/output_pd/cogspaces/full')


components = []
snrs = []
regex = re.compile(r'[0-9]+$')
for this_dir in filter(regex.match, os.listdir(output_dir)):
    this_exp_dir = join(output_dir, this_dir)
    component = join(this_exp_dir, 'maps', 'components.nii.gz')
    snr = join(this_exp_dir, 'maps', 'snr.nii.gz')
    if os.path.exists(component) and os.path.exists(snr):
        components.append(component)
        snrs.append(snr)

mask = fetch_mask()['hcp']

masker = MultiNiftiMasker(mask_img=mask, memory=mem, n_jobs=10, memory_level=1).fit()

components = masker.transform(components[:3])

stabilities = mem.cache(compute_stabilities)(components)
stabilities = np.mean(stabilities, axis=2)
max_stab = np.max(stabilities)
min_stab = np.min(stabilities)
print(max_stab, min_stab)