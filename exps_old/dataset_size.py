from os.path import join

from cogspaces.pipeline import get_output_dir, make_data_frame

reduced_dir = join(get_output_dir(), 'reduced')
unmask_dir = join(get_output_dir(), 'unmasked')

datasets = ['archi', 'brainomics', 'camcan', 'la5c']

df = make_data_frame(datasets,
                     'hcp_rs_positive',
                     reduced_dir=reduced_dir,
                     unmask_dir=unmask_dir)


for dataset in datasets:
    print(len(df.loc[dataset].index.get_level_values('subject').unique()))