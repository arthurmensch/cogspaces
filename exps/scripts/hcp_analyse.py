import numpy as np
import os
from nilearn.input_data import NiftiMasker
from os.path import expanduser, join

from cogspaces.datasets.utils import fetch_mask

output_dir = expanduser('~/output/modl/components/hcp/')

mask = fetch_mask()['hcp']
masker = NiftiMasker(mask_img=mask).fit()

for dirname, subdirs, filenames in os.walk(output_dir):
    for filename in filter(lambda x: 'components.nii.gz' in x, filenames):
        filename = join(dirname, filename)
        print(filename)
        components = masker.transform(filename)
        print('n components', components.shape[0])
        mean_overlap = np.sum(components != 0, axis=0).mean()
        zeros = np.all(components == 0, axis=0).sum()
        print('Mean overlap', mean_overlap)
        print('Zeros', zeros)