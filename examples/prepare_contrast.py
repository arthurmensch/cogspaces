from os.path import join

from cogspaces.input_data.fixes import monkey_patch_nifti_image

from cogspaces import get_output_dir
from cogspaces.input_data import unmask, reduce

monkey_patch_nifti_image()

output_dir = join(get_output_dir()[0], 'unmask')

n_jobs = 10
batch_size = 1200

for dataset in ['hcp', 'archi', 'human_voice', 'brainomics', 'la5c', 'camcan']:
    unmask(dataset, output_dir=output_dir,
           n_jobs=n_jobs, batch_size=batch_size)
    reduce(dataset, output_dir=output_dir)
