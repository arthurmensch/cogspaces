import glob
import os
import re
from os.path import join

import pandas as pd
from modl.datasets import get_data_dirs


def fetch_camcan(data_dir=None, n_subjects=None):
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'camcan', 'camcan_smt_maps')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that archi can be found under $MODL_DATA'
            'repository.')
    z_maps = glob.glob(join(source_dir, '*', '*_z_score.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    directions = []
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
            directions.append('level1')
    df = pd.DataFrame(data={'subject': subjects,
                            'task': tasks,
                            'contrast': contrasts,
                            'direction': directions,
                            'z_map': filtered_z_maps, })
    df.set_index(['subject', 'task', 'contrast', 'direction'], inplace=True)
    df.sort_index(inplace=True)
    subjects = df.index.get_level_values('subject').unique().values.tolist()
    df = df.loc[subjects[:n_subjects]]
    return df


def fetch_brainomics(data_dir=None, n_subjects=None):
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'brainomics')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that archi can be found under $MODL_DATA'
            'repository.')
    z_maps = glob.glob(join(source_dir, '*', 'c_*.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    directions = []
    regex = re.compile('.*vs.*')
    for z_map in z_maps:
        match = re.match(regex, z_map)
        if match is None:
            dirname, contrast = os.path.split(z_map)
            contrast = contrast[6:-7]
            subject = int(dirname[-2:])
            subjects.append(subject)
            contrasts.append(contrast)
            tasks.append('localizer')
            filtered_z_maps.append(z_map)
            directions.append('level1')
    df = pd.DataFrame(data={'subject': subjects,
                            'task': tasks,
                            'contrast': contrasts,
                            'direction': directions,
                            'z_map': filtered_z_maps, })
    df.set_index(['subject', 'task', 'contrast', 'direction'], inplace=True)
    df.sort_index(inplace=True)
    subjects = df.index.get_level_values('subject').unique().values.tolist()
    df = df.loc[subjects[:n_subjects]]
    return df


def fetch_archi(data_dir=None, n_subjects=None):
    INTERESTING_CONTRASTS_EXTENDED = ["expression_control",
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

    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'archi', 'glm')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that archi can be found under $MODL_DATA'
            'repository.')
    z_maps = glob.glob(join(source_dir, '*/*/*', 'z_*.nii.gz'))
    subjects = []
    contrasts = []
    tasks = []
    filtered_z_maps = []
    directions = []
    for z_map in z_maps:
        dirname, contrast = os.path.split(z_map)
        contrast = contrast[2:-7]
        if contrast in INTERESTING_CONTRASTS_EXTENDED:
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
                            'direction': directions,
                            'z_map': filtered_z_maps, })
    df.set_index(['subject', 'task', 'contrast', 'direction'], inplace=True)
    df.sort_index(inplace=True)
    subjects = df.index.get_level_values('subject').unique().values.tolist()
    df = df.loc[subjects[:n_subjects]]
    return df


def fetch_human_voice(data_dir=None, n_subjects=None):
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'human_voice', 'ds000158_R1.0.1', 'glm')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that human_voice can be found under $MODL_DATA'
            'repository.')
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
                            'direction': directions,
                            'z_map': filtered_z_maps, })
    df.set_index(['subject', 'task', 'contrast', 'direction'], inplace=True)
    df.sort_index(inplace=True)
    subjects = df.index.get_level_values('subject').unique().values.tolist()
    df = df.loc[subjects[:n_subjects]]
    return df


def fetch_la5c(data_dir=None, n_subjects=None):
    data_dir = get_data_dirs(data_dir)[0]
    source_dir = join(data_dir, 'la5c', 'ds000030', 'glm')
    if not os.path.exists(source_dir):
        raise ValueError(
            'Please ensure that archi can be found under $MODL_DATA'
            'repository.')
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
                            'direction': directions,
                            'z_map': filtered_z_maps, })
    df.set_index(['subject', 'task', 'contrast', 'direction'], inplace=True)
    df.sort_index(inplace=True)
    subjects = df.index.get_level_values('subject').unique().values.tolist()
    df = df.loc[subjects[:n_subjects]]
    return df