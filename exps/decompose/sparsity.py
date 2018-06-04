import numpy as np
from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask
from nilearn.input_data import NiftiMasker

mask = fetch_mask()
modl_atlas = fetch_atlas_modl()
masker = NiftiMasker(mask_img=mask).fit()

for name, dictionary in modl_atlas.items():
    components = masker.transform(dictionary)
    print(name)
    nnz = np.sum(components.sum(axis=0) != 0)
    print(nnz)
