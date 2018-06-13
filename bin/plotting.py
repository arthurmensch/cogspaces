import argparse
import os
from jinja2 import Template
from os.path import join, expanduser, abspath

from cogspaces.plotting import plot_all

path = abspath(__file__)

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
dirname, filename = os.path.split(filename)
assert filename.endswith('.nii.gz'), ValueError('Wrong file argument')
if output == '':
    output_dir = join(dirname, filename.replace('.nii.gz', ''))
else:
    output_dir = expanduser(output)
print(output_dir)
    
imgs = plot_all(filename, output_dir=output_dir, n_jobs=n_jobs)


template = join(os.path.dirname(path), 'plotting.html')

with open(template, 'r') as f:
    template = f.read()
template = Template(template)
html = template.render(imgs=imgs)
html_name = filename.replace('.nii.gz', '.html')
with open(html_name, 'w+') as f:
    f.write(html)
