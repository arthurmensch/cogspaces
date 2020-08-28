import pandas as pd

from cogspaces.datasets.derivative import get_study_info

info = get_study_info().groupby(by='study').first()

info = info[['latex_name', '#contrasts_per_study', '#subjects']].reset_index(drop=True)
total = info.sum(axis=0)
total['latex_name'] = 'Total'
total = total.to_frame().T
info = pd.concat([info, total], axis=0)
info.columns = pd.Index(['Study and task description', '\# contrasts', '\# subjects'])
with pd.option_context("max_colwidth", 1000):
    latex = info.to_latex(index=False, escape=False)
with open('table1.tex', 'w+') as f:
    f.write(latex)


# info['table_name'] = info.apply(lambda x: f'\cite{{{x["citekey"]}}} {x["comment"]}', axis='columns')
