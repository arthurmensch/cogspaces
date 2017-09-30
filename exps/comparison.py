import json
from os.path import join
import os

import pandas as pd

from cogspaces.pipeline import get_output_dir

from json import JSONDecodeError

# 11 factored
# 12 aborted logistic
# 24 all best standardization   |  logistic dropout to rerun
# 28 no cross val on dropout (good)   |
# 30 Last one ?
basedir_ids = [30]
basedirs = [join(get_output_dir(), 'multi_nested', str(_id), 'run') for _id in basedir_ids]
res_list = []
for basedir in basedirs:
    for exp_dir in os.listdir(basedir):
        exp_dir = join(basedir, exp_dir)
        try:
            config = json.load(open(join(exp_dir, 'config.json'), 'r'))
            info = json.load(open(join(exp_dir, 'info.json'), 'r'))
        except (JSONDecodeError, FileNotFoundError):
            continue
        datasets = config['datasets']
        datasets = '__'.join(datasets)
        config['datasets'] = datasets
        score = info.pop('score')
        res = dict(**config, **info)
        for key, value in score.items():
            res[key] = value
        res_list.append(res)
res = pd.DataFrame(res_list)

df_agg = res.groupby(by=['datasets', 'model', 'with_std', 'source']).aggregate(['mean', 'std', 'count'])

df_agg = df_agg.fillna(0)

results = {}
for dataset in ['archi', 'brainomics', 'camcan', 'la5c']:
    results[dataset] = df_agg.loc[[dataset, '%s__hcp' % dataset]]['test_%s' % dataset]

results = pd.concat(results)
print(results)
results.to_csv('results.csv')
