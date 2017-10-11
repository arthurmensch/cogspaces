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
basedir_ids = [31]
basedirs = [join(get_output_dir(), 'multi_nested', str(_id), 'run') for _id in basedir_ids]
basedir_ids = [6]
basedirs += [join(get_output_dir(), 'benchmark', str(_id), 'run') for _id in basedir_ids]
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
        dataset = datasets[0]
        if len(datasets) > 1:
            helper_datasets = '__'.join(datasets[1:])
        else:
            helper_datasets = 'none'
        config['dataset'] = dataset
        config['helper_datasets'] = helper_datasets
        score = info.pop('score')
        res = dict(**config, **info)
        for key, value in score.items():
            res[key] = value
        res_list.append(res)
res = pd.DataFrame(res_list)

df_agg = res.groupby(by=['dataset', 'source', 'model', 'with_std',
                         'helper_datasets']).aggregate(['mean', 'std', 'count'])

df_agg = df_agg.fillna(0)

results = {}
for dataset in ['archi', 'brainomics', 'camcan', 'la5c']:
    results[dataset] = df_agg.loc[dataset]['test_%s' % dataset]

results = pd.concat(results, names=['dataset'])
print(results)
results.to_csv('results.csv')
