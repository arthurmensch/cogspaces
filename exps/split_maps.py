from os.path import join, expanduser

import nibabel
import nilearn
from nilearn.input_data import MultiNiftiMasker, NiftiMasker

from cogspaces.datasets.utils import get_output_dir

file = join(get_output_dir(), 'final_latent', 'components.nii.gz')
# masker = NiftiMasker(smoothing_fwhm=None, standardize=False).fit(file)
# imgs = masker.transform(file)

for i, img in enumerate(nilearn.image.iter_img(file)):
    nibabel.save(img, expanduser(f'~/components_{i}.nii.gz'))