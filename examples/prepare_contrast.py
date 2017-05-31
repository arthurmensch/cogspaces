from os.path import join

from cogspaces.input_data.fixes import monkey_patch_nifti_image

from cogspaces.utils import get_output_dir
from cogspaces.input_data.base import unmask, reduce

monkey_patch_nifti_image()

output_dir = get_output_dir()

n_jobs = 10
batch_size = 1200

for dataset in ['archi', 'hcp', 'camcan', 'brainomics', 'la5c']:
    # unmask(dataset, output_dir=output_dir,
    #        n_jobs=n_jobs, batch_size=batch_size)
    reduce(dataset, output_dir=output_dir, source='hcp_rs')
