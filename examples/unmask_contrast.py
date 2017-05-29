from os.path import join

from modl.input_data.fmri.monkey import monkey_patch_nifti_image

from cogspaces import get_output_dir
from cogspaces.input_data import unmask

monkey_patch_nifti_image()

output_dir = join(get_output_dir()[0], 'unmask')

n_jobs = 10
batch_size = 1200
dataset = 'human_voice'

unmask('human_voice', output_dir=output_dir,
       n_jobs=n_jobs, batch_size=batch_size)
