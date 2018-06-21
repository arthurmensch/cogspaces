import json
import numpy as np
import os
import re
from jinja2 import Template
from joblib import load, dump
from nilearn._utils import check_niimg
from os.path import join

from cogspaces.datasets.utils import get_output_dir
from cogspaces.plotting import plot_all
from cogspaces.utils import get_dictionary, get_masker

# output_dir = join(get_output_dir(), 'full_logistic_full')
# full = True

output_dir = join(get_output_dir(), 'logistic_gm_full')
full = False


def classifs_html(output_dir, classifs_dir):
    with open('plot_maps.html', 'r') as f:
        template = f.read()
    names = load(join(output_dir, 'names.pkl'))
    template = Template(template)
    imgs = []
    for name in names:
        view_types = ['stat_map', 'glass_brain',
                      ]
        srcs = []
        for view_type in view_types:
            src = join(classifs_dir, '%s_%s.png' % (name, view_type))
            srcs.append(src)
        imgs.append((srcs, name))
    html = template.render(imgs=imgs)
    output_file = join(output_dir, 'classifs.html')
    with open(output_file, 'w+') as f:
        f.write(html)

regex = re.compile(r'[0-9]+$')
res = []
names = []
coefs = []

for this_dir in filter(regex.match, os.listdir(output_dir)):
    this_exp_dir = join(output_dir, this_dir)
    this_dir = int(this_dir)
    try:
        config = json.load(
            open(join(this_exp_dir, 'config.json'), 'r'))
        run = json.load(open(join(this_exp_dir, 'run.json'), 'r'))
        study = config['data']['studies']
        target_encoder = load(join(this_exp_dir, 'target_encoder.pkl'), 'r')
        estimator = load(join(this_exp_dir, 'estimator.pkl'), 'r')
    except FileNotFoundError:
        continue
    contrasts = target_encoder.le_[study]['contrast'].classes_
    coefs.append(estimator.coef_[study])
    names.extend(['%s::%s' % (study, contrast) for contrast in contrasts])
names = np.array(names)
coefs = np.concatenate(coefs, axis=0)
sort = np.argsort(names)
names = names[sort].tolist()
coefs = coefs[sort]

dictionary = get_dictionary()
if not full:
    components = coefs.dot(dictionary)
else:
    mask = np.any(dictionary, axis=0)
    components = np.zeros((coefs.shape[0], dictionary.shape[1]))
    components[:, mask] = coefs

masker = get_masker()
img = masker.inverse_transform(components)

img.to_filename(join(output_dir, 'components.nii.gz'))
dump(names, join(output_dir, 'names.pkl'))

#
# ###############################################################################
# names = load(join(output_dir, 'names.pkl'))
# img = check_niimg(join(output_dir, 'components.nii.gz'))
#
# plot_all(img, names=names, threshold=0,
#          output_dir=join(output_dir, 'classifs'),
#          view_types=['stat_map', 'glass_brain'], n_jobs=40)
# classifs_html(output_dir, 'classifs')
