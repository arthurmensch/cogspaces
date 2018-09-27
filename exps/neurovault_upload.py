from cogspaces.datasets.contrasts import fetch_all, fetch_contrasts
import pandas as pd
from pynv import Client

cogatlas = {'EMOTION': 'trm_550b5b066d37b',
            'GAMBLING': 'trm_550b5c1a7f4db',
            'LANGUAGE': 'trm_550b54a8b30f4',
            'MOTOR': 'trm_550b53d7dd674',
            'RELATIONAL': 'trm_550b5a47aa23e',
            'SOCIAL': 'trm_550b557e5f90e',
            'WM': 'trm_550b50095d4a3'
           }


data_dir = '/storage/store/data/cogspaces'
dfs = []
for study in ['archi', 'brainomics', 'camcan', 'hcp',
              'la5c', 'brainpedia']:
    df = fetch_contrasts(study, data_dir=data_dir)
    dfs.append(df)

df = pd.concat(dfs)
df = df.reset_index()

api = Client(access_token='FqLlQV6L7W382QYWK5GRdA0vu7NjoBRRtNxK8yLA')

collection = api.create_collection('HCP 900')

hcp_df = df[df['study'] == ' hcp']


image = api.add_image(
    collection['id'],
    image_file_path,
    name='fMRI_BOLD',
    modality='Other',
    map_type='Z'
)