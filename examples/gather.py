import json
from os.path import join

import pandas as pd

from cogspaces.utils import get_output_dir

basedir = join(get_output_dir(), 'multi')
res = []
for i in range(100):
    exp_dir = join(basedir, str(i))
    config = json.read(open(join(exp_dir, 'config.json'), 'r'))
    info = json.read(open(join(exp_dir, 'info.json'), 'r'))
    datasets = config['datasets']
    datasets = '__'.join(datasets)
    alpha = config['alpha']
    score = config['score']
    res = {'datasets': datasets, 'alpha': alpha}
    for key, value in score.items():
        res[key] = value
res = pd.DataFrame(res)
