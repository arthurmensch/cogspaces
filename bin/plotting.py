import argparse
import os
from os.path import join, expanduser

from cogspaces.plotting import plot_all

parser = argparse.ArgumentParser(
    description='Convert a Nifti image to an html file with'
                'pictures.')
parser.add_argument('filename', metavar='filename', type=str,
                    help='Name of the Nifti file')
parser.add_argument('--output', dest='output', type=str,
                    default='',
                    help='output directory (default: same name as the Nifti file)')
parser.add_argument('--n_jobs', dest='n_jobs', type=int,
                    default=1,
                    help='Number of jobs to use')

args = parser.parse_args()
filename = args.filename
output = args.output
n_jobs = args.n_jobs
dirname, tailname = os.path.split(filename)
assert tailname[-7:] == '.nii.gz', ValueError('Wrong file argument')
tailname = tailname[:-7]
if output == '':
    output_dir = join(dirname, tailname)
else:
    output_dir = expanduser(output)
plot_all(filename, output_dir=output_dir, n_jobs=n_jobs)