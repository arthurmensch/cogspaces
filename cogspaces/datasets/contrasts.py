import glob
import os
import re
from os.path import join

import pandas as pd
from cogspaces.datasets.utils import get_data_dir
from hcp_builder.dataset import fetch_hcp as hcp_builder_fetch_hcp
from sklearn.utils import Bunch

idx = pd.IndexSlice


def replace_filename_unmasked(img):
    return img.replace('.nii.gz', '.npy').replace('cogspaces',
                                                  'cogspaces/unmasked')


def fetch_contrasts(study, data_dir=None):
    if study == 'archi':
        return fetch_archi(data_dir=data_dir)
    elif study == 'brainomics':
        return fetch_brainomics(data_dir=data_dir)
    elif study == 'la5c':
        return fetch_la5c(data_dir=data_dir)
    elif study == 'camcan':
        return fetch_camcan(data_dir=data_dir)
    elif study == 'hcp':
        return fetch_hcp(data_dir=data_dir).contrasts
    elif study == 'brainpedia':
        return fetch_brainpedia(data_dir=data_dir)
    else:
        raise ValueError


def fetch_all(data_dir=None, unmasked=False):
    dfs = []
    for study in ['archi', 'brainomics', 'camcan', 'hcp',
                    'la5c', 'brainpedia']:
        df = fetch_contrasts(study, data_dir=data_dir)
        dfs.append(df)
    df = pd.concat(dfs)
    if unmasked:
        df['z_map'] = df['z_map'].map(replace_filename_unmasked)
    return df


def fetch_camcan(data_dir=None):
    data_dir = get_data_dir(data_dir)
    source_dir = join(data_dir, 'camcan', 'camcan_smt_maps')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that %s contains all required data.'
            % source_dir)
    z_maps = glob.glob(join(source_dir, '*', '*_z_score.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    for z_map in z_maps:
        dirname, contrast = os.path.split(z_map)
        _, dirname = os.path.split(dirname)
        contrast = contrast[13:-15]
        subject = int(dirname[6:])
        if contrast in ['AudOnly', 'VidOnly', 'AudVid1200',
                        'AudVid300', 'AudVid600']:
            subjects.append(subject)
            contrasts.append(contrast)
            if contrast in ['AudOnly', 'VidOnly']:
                tasks.append('audio-video')
            else:
                tasks.append('AV-freq')
            filtered_z_maps.append(z_map)
    df = pd.DataFrame(data={'subject': subjects,
                            'task': tasks,
                            'contrast': contrasts,
                            'direction': 'level1',
                            'study': 'camcan',
                            'z_map': filtered_z_maps, })
    df.set_index(['study', 'subject', 'task', 'contrast', 'direction'],
                 inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_brainomics(data_dir=None):
    data_dir = get_data_dir(data_dir)
    source_dir = join(data_dir, 'brainomics')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that %s contains all required data.' % source_dir)
    z_maps = glob.glob(join(source_dir, '*', 'c_*.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    regex = re.compile('.*vs.*')
    for z_map in z_maps:
        match = re.match(regex, z_map)
        if match is None and z_map != 'effects_of_interest':
            dirname, contrast = os.path.split(z_map)
            contrast = contrast[6:-7]
            subject = int(dirname[-2:])
            subjects.append(subject)
            contrasts.append(contrast)
            tasks.append('localizer')
            filtered_z_maps.append(z_map)
    df = pd.DataFrame(data={'subject': subjects,
                            'task': tasks,
                            'contrast': contrasts,
                            'direction': 'level1',
                            'study': 'brainomics',
                            'z_map': filtered_z_maps, })
    df.set_index(['study', 'subject', 'task',
                  'contrast', 'direction'], inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_archi(data_dir=None):
    INTERESTING_CONTRASTS = ["expression_control",
                             "expression_intention",
                             "expression_sex", "face_control",
                             "face_sex",
                             "face_trusty", "audio", "calculaudio",
                             "calculvideo",
                             "clicDaudio", "clicDvideo", "clicGaudio",
                             "clicGvideo", "computation",
                             "damier_H", "damier_V", "object_grasp",
                             "object_orientation", "rotation_hand",
                             "rotation_side", "saccade",
                             "motor-cognitive",
                             "false_belief_audio",
                             "false_belief_video",
                             "mecanistic_audio", "mecanistic_video",
                             "non_speech", "speech",
                             "triangle_intention",
                             "triangle_random"]

    data_dir = get_data_dir(data_dir)
    source_dir = join(data_dir, 'archi', 'glm')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that %s contains all required data.' % source_dir)
    z_maps = glob.glob(join(source_dir, '*/*/*', 'z_*.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    for z_map in z_maps:
        dirname, contrast = os.path.split(z_map)
        contrast = contrast[2:-7]
        if contrast in INTERESTING_CONTRASTS:
            dirname, _ = os.path.split(dirname)
            dirname, task = os.path.split(dirname)
            dirname, subject = os.path.split(dirname)
            subject = int(subject[-3:])
            subjects.append(subject)
            contrasts.append(contrast)
            tasks.append(task)
            filtered_z_maps.append(z_map)
    df = pd.DataFrame(data={'subject': subjects,
                            'task': tasks,
                            'contrast': contrasts,
                            'direction': 'level1',
                            'study': 'archi',
                            'z_map': filtered_z_maps, })
    df.set_index(['study', 'subject',
                  'task', 'contrast', 'direction'], inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_human_voice(data_dir=None):
    data_dir = get_data_dir(data_dir)
    source_dir = join(data_dir,
                      'human_voice', 'ds000158_R1.0.1', 'glm')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that %s contains all required data.' % source_dir)
    z_maps = glob.glob(join(source_dir, '*/*/*', 'z_*.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    directions = []
    for z_map in z_maps:
        dirname, contrast = os.path.split(z_map)
        contrast = contrast[2:-7]
        dirname, _ = os.path.split(dirname)
        dirname, task = os.path.split(dirname)
        dirname, subject = os.path.split(dirname)
        subject = int(subject[-3:])
        subjects.append(subject)
        contrasts.append(contrast)
        tasks.append(task)
        filtered_z_maps.append(z_map)
        directions.append('level1')
    df = pd.DataFrame(data={'subject': subjects,
                            'task': tasks,
                            'contrast': contrasts,
                            'direction': 'level1',
                            'study': 'human_voice',
                            'z_map': filtered_z_maps, })
    df.set_index(['subject', 'task', 'contrast', 'direction'], inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_la5c(data_dir=None):
    data_dir = get_data_dir(data_dir)
    source_dir = join(data_dir,
                      'la5c', 'ds000030', 'glm')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that %s contains all required data.'
            % source_dir)
    z_maps = glob.glob(join(source_dir, '*/*/*', 'z_*.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    for z_map in z_maps:
        dirname, contrast = os.path.split(z_map)
        contrast = contrast[2:-7]
        dirname, _ = os.path.split(dirname)
        dirname, task = os.path.split(dirname)
        dirname, subject = os.path.split(dirname)
        subject = int(subject[-3:])
        subjects.append(subject)
        contrasts.append(contrast)
        tasks.append(task)
        filtered_z_maps.append(z_map)
    df = pd.DataFrame(data={'study': 'la5c',
                            'subject': subjects,
                            'task': tasks,
                            'contrast': contrasts,
                            'direction': 'level1',
                            'z_map': filtered_z_maps, })
    df.set_index(['study', 'subject', 'task', 'contrast', 'direction'],
                 inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_brainpedia(data_dir=None, drop_some=True):
    data_dir = get_data_dir(data_dir)
    source_dir = join(data_dir,  'brainpedia')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that %s contains all required data.'
            % source_dir)
    studies = os.listdir(source_dir)
    rec = []
    for study in studies:
        if 'archi' in study:
            continue
        study_dir = join(source_dir, study)
        subjects = os.listdir(study_dir)
        for subject in subjects:
            if subject == 'models':
                continue
            subject_dir = join(study_dir, subject, 'model', 'model002',
                               'z_maps')
            subject = int(subject[3:])
            maps = os.listdir(subject_dir)
            for this_map in maps:
                task = int(this_map[4:7])
                contrast = this_map[8:-7]
                rec.append({'study': study,
                            'subject': subject,
                            'task': task,
                            'direction': 'level2',
                            'contrast': contrast,
                            'z_map': join(subject_dir, this_map)})
    df = pd.DataFrame(rec)
    df.set_index(['study', 'subject', 'task', 'contrast', 'direction'],
                 inplace=True)
    if drop_some:
        df.drop('pinel2007fast', level=0, axis=0, inplace=True)
        df.drop('ds102', level=0, axis=0, inplace=True)
    return df


def fetch_hcp(data_dir=None, n_subjects=None, subjects=None,
              from_file=True):
    data_dir = get_data_dir(data_dir)
    BASE_CONTRASTS = ['FACES', 'SHAPES', 'PUNISH', 'REWARD',
                      'MATH', 'STORY', 'MATCH', 'REL',
                      'RANDOM', 'TOM',
                      'LF', 'RF', 'LH', 'RH', 'CUE',
                      '0BK_BODY', '0BK_FACE', '0BK_PLACE',
                      '0BK_TOOL',
                      '2BK_BODY', '2BK_FACE', '2BK_PLACE',
                      '2BK_TOOL',
                      ]

    source_dir = join(data_dir, 'HCP900')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that %s contains all required data.'
            % source_dir)
    res = hcp_builder_fetch_hcp(data_dir=source_dir, n_subjects=n_subjects,
                                from_file=from_file,
                                subjects=subjects, on_disk=True)
    rest = res.rest.assign(confounds=[None] * res.rest.shape[0])
    task = res.task.assign(confounds=[None] * res.task.shape[0])

    task.sort_index(inplace=True)
    rest.sort_index(inplace=True)

    # Make it compatible with the other studies
    contrasts = res.contrasts.loc[idx[:, :, BASE_CONTRASTS, :], :]
    contrasts = contrasts[['z_map']]
    contrasts.reset_index(inplace=True)
    contrasts['study'] = 'hcp'
    contrasts.set_index(
        ['study', 'subject', 'task', 'contrast', 'direction'],
        inplace=True)
    contrasts.sort_index(inplace=True)
    return Bunch(rest=rest,
                 contrasts=contrasts,
                 task=task,
                 behavioral=res.behavioral,
                 mask=res.mask,
                 root=res.root)