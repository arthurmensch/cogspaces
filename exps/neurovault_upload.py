import os
from math import ceil

import pandas as pd
from joblib import Parallel, delayed
from pynv import Client
from sklearn.utils import gen_batches

from cogspaces.datasets.contrasts import fetch_contrasts

access_token = os.environ['NEUROVAULT_ACCESS_TOKEN']

data = {'hcp': {'name': 'HCP 900',
                'cogatlas': {'EMOTION': 'trm_550b5b066d37b',
                             'GAMBLING': 'trm_550b5c1a7f4db',
                             'LANGUAGE': 'trm_550b54a8b30f4',
                             'MOTOR': 'trm_550b53d7dd674',
                             'RELATIONAL': 'trm_550b5a47aa23e',
                             'SOCIAL': 'trm_550b557e5f90e',
                             'WM': 'trm_550b50095d4a3'
                             }},
        'archi': {'name': 'Archi',
                  'cogatlas': {'emotional': 'trm_551efd8a98162',
                               'localizer': 'trm_553e85265f51e',
                               }},
        'brainomics': {'name': 'Brainomics',
                       'cogatlas': {}},
        'camcan': {'name': 'Camcan sensori-motor task',
                   'cogatlas': {}},
        'la5c': {'name': 'UCLA Consortium for Neuropsychiatric Phenomics LA5c Study',
                   'cogatlas': {}}
        }

data_dir = '/storage/store/data/cogspaces'
dfs = []
for study in ['archi', 'brainomics', 'camcan', 'hcp',
              'la5c', 'brainpedia']:
    df = fetch_contrasts(study, data_dir=data_dir)
    dfs.append(df)

df = pd.concat(dfs)
df = df.reset_index()


def upload(this_df, collection_id, cogatlas=None):
    api = Client(access_token=access_token)
    collection = api.get_collection(collection_id)
    mock = False

    total_length = len(this_df)
    for i, (index, elem) in enumerate(this_df.iterrows()):
        path = elem['z_map']
        task = elem['task']
        subject = elem['subject']
        contrast = elem['contrast']
        if cogatlas is not None and task in cogatlas:
            data = dict(cognitive_paradigm_cogatlas=cogatlas[task])
        else:
            data = {}
        name = f'{subject}_{task}_{contrast}'
        print(f'Upload {i}/{total_length} path {path}, paradigm {task},'
              f' contrast {contrast}, data {data}')
        if mock:
            pass
        else:
            image = api.add_image(
                collection['id'],
                path,
                modality='fMRI-BOLD',
                name=name,
                number_of_subjects=1,
                target_template_image='MNI152NLin2009cAsym',
                contrast_definition=contrast,
                task=task,
                subject=subject,
                map_type='Z',
                **data
            )


api = Client(access_token=access_token)

for study in ['archi', 'brainomics', 'camcan', 'la5c']:
    this_df = df.loc[df['study'] == study]
    n_jobs = 10
    total_length = len(this_df)
    collection = api.create_collection(data[study]['name'])
    Parallel(n_jobs=n_jobs)(
        delayed(upload)(this_df.iloc[batch], collection['id'],
                        cogatlas=data[study]['cogatlas'])
        for batch in gen_batches(total_length,
                                 int(ceil(total_length
                                          / n_jobs))))
