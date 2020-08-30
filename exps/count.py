import os
from os.path import join

import pandas as pd

from cogspaces.datasets.derivative import get_study_info
from cogspaces.datasets.utils import get_output_dir

info = get_study_info().groupby(by='study').first()

info = info[['latex_name_study', '#contrasts_per_study', '#subjects']].reset_index(drop=True)
total = info.sum(axis=0)
total['latex_name_study'] = 'Total'
total = total.to_frame().T
info = pd.concat([info, total], axis=0)
info.columns = pd.Index(['Study and task description', '\# contrasts', '\# subjects'])
with pd.option_context("max_colwidth", 1000):
    latex = info.to_latex(index=False, escape=False, column_format='p{9cm}ll')

table_dir = join(get_output_dir(), 'revision_output')
if not os.path.exists(table_dir):
    os.makedirs(table_dir)

with open(join(table_dir, 'count.tex'), 'w+') as f:
    f.write(latex)


# info['table_name'] = info.apply(lambda x: f'\cite{{{x["citekey"]}}} {x["comment"]}', axis='columns')
