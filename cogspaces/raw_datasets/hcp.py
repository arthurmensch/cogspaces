import os
import sys
import traceback
import warnings
from os.path import join

import boto
import nibabel
import numpy as np
import pandas as pd
from boto.s3.key import Key
from nilearn.datasets.utils import _fetch_file
from sklearn.datasets.base import Bunch

TASK_LIST = ['EMOTION', 'WM', 'MOTOR', 'RELATIONAL',
             'GAMBLING', 'SOCIAL', 'LANGUAGE']

EVS = {'EMOTION': {'EMOTION_Stats.csv',
                   'Sync.txt',
                   'fear.txt',
                   'neut.txt'},
       'GAMBLING': {
           'GAMBLING_Stats.csv',
           'Sync.txt',
           'loss.txt',
           'loss_event.txt',
           'neut_event.txt',
           'win.txt',
           'win_event.txt',
       },
       'LANGUAGE': {
           'LANGUAGE_Stats.csv',
           'Sync.txt',
           'cue.txt',
           'math.txt',
           'present_math.txt',
           'present_story.txt',
           'question_math.txt',
           'question_story.txt',
           'response_math.txt',
           'response_story.txt',
           'story.txt',
       },
       'MOTOR': {
           'Sync.txt',
           'cue.txt',
           'lf.txt',
           'lh.txt',
           'rf.txt',
           'rh.txt',
           't.txt',
       },
       'RELATIONAL': {
           'RELATIONAL_Stats.csv',
           'Sync.txt',
           'error.txt',
           'match.txt',
           'relation.txt',
       },
       'SOCIAL': {
           'SOCIAL_Stats.csv',
           'Sync.txt',
           'mental.txt',
           'mental_resp.txt',
           'other_resp.txt',
           'rnd.txt',
       },
       'WM': {
           '0bk_body.txt',
           '0bk_cor.txt',
           '0bk_err.txt',
           '0bk_faces.txt',
           '0bk_nlr.txt',
           '0bk_places.txt',
           '0bk_tools.txt',
           '2bk_body.txt',
           '2bk_cor.txt',
           '2bk_err.txt',
           '2bk_faces.txt',
           '2bk_nlr.txt',
           '2bk_places.txt',
           '2bk_tools.txt',
           'Sync.txt',
           'WM_Stats.csv',
           'all_bk_cor.txt',
           'all_bk_err.txt'}
       }

CONTRASTS = [["WM", 1, "2BK_BODY"],
             ["WM", 2, "2BK_FACE"],
             ["WM", 3, "2BK_PLACE"],
             ["WM", 4, "2BK_TOOL"],
             ["WM", 5, "0BK_BODY"],
             ["WM", 6, "0BK_FACE"],
             ["WM", 7, "0BK_PLACE"],
             ["WM", 8, "0BK_TOOL"],
             ["WM", 9, "2BK"],
             ["WM", 10, "0BK"],
             ["WM", 11, "2BK-0BK"],
             ["WM", 12, "neg_2BK"],
             ["WM", 13, "neg_0BK"],
             ["WM", 14, "0BK-2BK"],
             ["WM", 15, "BODY"],
             ["WM", 16, "FACE"],
             ["WM", 17, "PLACE"],
             ["WM", 18, "TOOL"],
             ["WM", 19, "BODY-AVG"],
             ["WM", 20, "FACE-AVG"],
             ["WM", 21, "PLACE-AVG"],
             ["WM", 22, "TOOL-AVG"],
             ["WM", 23, "neg_BODY"],
             ["WM", 24, "neg_FACE"],
             ["WM", 25, "neg_PLACE"],
             ["WM", 26, "neg_TOOL"],
             ["WM", 27, "AVG-BODY"],
             ["WM", 28, "AVG-FACE"],
             ["WM", 29, "AVG-PLACE"],
             ["WM", 30, "AVG-TOOL"],
             ["GAMBLING", 1, "PUNISH"],
             ["GAMBLING", 2, "REWARD"],
             ["GAMBLING", 3, "PUNISH-REWARD"],
             ["GAMBLING", 4, "neg_PUNISH"],
             ["GAMBLING", 5, "neg_REWARD"],
             ["GAMBLING", 6, "REWARD-PUNISH"],
             ["MOTOR", 1, "CUE"],
             ["MOTOR", 2, "LF"],
             ["MOTOR", 3, "LH"],
             ["MOTOR", 4, "RF"],
             ["MOTOR", 5, "RH"],
             ["MOTOR", 6, "T"],
             ["MOTOR", 7, "AVG"],
             ["MOTOR", 8, "CUE-AVG"],
             ["MOTOR", 9, "LF-AVG"],
             ["MOTOR", 10, "LH-AVG"],
             ["MOTOR", 11, "RF-AVG"],
             ["MOTOR", 12, "RH-AVG"],
             ["MOTOR", 13, "T-AVG"],
             ["MOTOR", 14, "neg_CUE"],
             ["MOTOR", 15, "neg_LF"],
             ["MOTOR", 16, "neg_LH"],
             ["MOTOR", 17, "neg_RF"],
             ["MOTOR", 18, "neg_RH"],
             ["MOTOR", 19, "neg_T"],
             ["MOTOR", 20, "neg_AVG"],
             ["MOTOR", 21, "AVG-CUE"],
             ["MOTOR", 22, "AVG-LF"],
             ["MOTOR", 23, "AVG-LH"],
             ["MOTOR", 24, "AVG-RF"],
             ["MOTOR", 25, "AVG-RH"],
             ["MOTOR", 26, "AVG-T"],
             ["LANGUAGE", 1, "MATH"],
             ["LANGUAGE", 2, "STORY"],
             ["LANGUAGE", 3, "MATH-STORY"],
             ["LANGUAGE", 4, "STORY-MATH"],
             ["LANGUAGE", 5, "neg_MATH"],
             ["LANGUAGE", 6, "neg_STORY"],
             ["SOCIAL", 1, "RANDOM"],
             ["SOCIAL", 2, "TOM"],
             ["SOCIAL", 3, "RANDOM-TOM"],
             ["SOCIAL", 4, "neg_RANDOM"],
             ["SOCIAL", 5, "neg_TOM"],
             ["SOCIAL", 6, "TOM-RANDOM"],
             ["RELATIONAL", 1, "MATCH"],
             ["RELATIONAL", 2, "REL"],
             ["RELATIONAL", 3, "MATCH-REL"],
             ["RELATIONAL", 4, "REL-MATCH"],
             ["RELATIONAL", 5, "neg_MATCH"],
             ["RELATIONAL", 6, "neg_REL"],
             ["EMOTION", 1, "FACES"],
             ["EMOTION", 2, "SHAPES"],
             ["EMOTION", 3, "FACES-SHAPES"],
             ["EMOTION", 4, "neg_FACES"],
             ["EMOTION", 5, "neg_SHAPES"],
             ["EMOTION", 6, "SHAPES-FACES"]]


def _init_s3_connection(aws_key, aws_secret,
                        bucket_name,
                        host='s3.amazonaws.com'):
    com = boto.connect_s3(aws_key, aws_secret, host=host)
    bucket = com.get_bucket(bucket_name, validate=False)
    return bucket


def _convert_to_s3_target(filename, data_dir=None):
    data_dir = get_data_dirs(data_dir)[0]
    if data_dir in filename:
        filename = filename.replace(data_dir, '/HCP_900')
    return filename


def fetch_hcp_timeseries(data_dir=None,
                         subjects=None,
                         n_subjects=None,
                         data_type='rest',
                         sessions=None,
                         on_disk=True,
                         tasks=None):
    """Utility to download from s3"""
    data_dir = get_data_dirs(data_dir)[0]

    if data_type not in ['task', 'rest']:
        raise ValueError("Wrong data type. Expected 'rest' or 'task', got"
                         "%s" % data_type)

    if subjects is None:
        subjects = fetch_subject_list(data_dir=data_dir,
                                      n_subjects=n_subjects)
    elif not hasattr(subjects, '__iter__'):
        subjects = [subjects]
    if not set(fetch_subject_list(data_dir=
                                  data_dir)).issuperset(set(subjects)):
        raise ValueError('Wrong subjects.')

    res = []
    for subject in subjects:
        subject_dir = join(data_dir, str(subject), 'MNINonLinear', 'Results')
        if data_type is 'task':
            if tasks is None:
                sessions = TASK_LIST
            elif isinstance(tasks, str):
                sessions = [tasks]
            if not set(TASK_LIST).issuperset(set(sessions)):
                raise ValueError('Wrong tasks.')
        else:
            if sessions is None:
                sessions = [1, 2]
            elif isinstance(sessions, int):
                sessions = [sessions]
            if not set([1, 2]).issuperset(set(sessions)):
                raise ValueError('Wrong rest sessions.')
        for session in sessions:
            for direction in ['LR', 'RL']:
                if data_type == 'task':
                    task = session
                    root_filename = 'tfMRI_%s_%s' % (task, direction)
                else:
                    root_filename = 'rfMRI_REST%i_%s' % (session,
                                                         direction)
                root_dir = join(subject_dir, root_filename)
                filename = join(root_dir, root_filename + '.nii.gz')
                mask = join(root_dir, root_filename + '_SBRef.nii.gz')
                confounds = ['Movement_AbsoluteRMS_mean.txt',
                             'Movement_AbsoluteRMS.txt',
                             'Movement_Regressors_dt.txt',
                             'Movement_Regressors.txt',
                             'Movement_RelativeRMS_mean.txt',
                             'Movement_RelativeRMS.txt']
                res_dict = {'filename': filename, 'mask': mask}
                print(filename)
                # for i, confound in enumerate(confounds):
                #     res_dict['confound_%i' % i] = join(root_dir, confound)
                if data_type is 'task':
                    feat_file = join(root_dir,
                                     "tfMRI_%s_%s_hp200_s4_level1.fsf"
                                     % (task, direction))
                    res_dict['feat_file'] = feat_file
                    for i, ev in enumerate(EVS[task]):
                        res_dict['ev_%i' % i] = join(root_dir, 'EVs', ev)
                requested_on_disk = os.path.exists(filename)
                res_dict['subject'] = subject
                res_dict['direction'] = direction
                if data_type == 'rest':
                    res_dict['session'] = session
                else:
                    res_dict['task'] = task
                if not on_disk or requested_on_disk:
                    res.append(res_dict)

    res = pd.DataFrame(res)
    if not res.empty:
        if data_type == 'rest':
            res.set_index(['subject', 'session', 'direction'],
                          inplace=True)
        else:
            res.set_index(['subject', 'task', 'direction'],
                          inplace=True)
    return res


def fetch_hcp_contrasts(data_dir=None,
                        output='nistats',
                        n_subjects=None,
                        subjects=None,
                        on_disk=True,
                        level=2):
    """Nilearn like fetcher"""
    data_dir = get_data_dirs(data_dir)[0]

    if subjects is None:
        subjects = fetch_subject_list(data_dir=data_dir,
                                      n_subjects=n_subjects)
    elif not hasattr(subjects, '__iter__'):
        subjects = [subjects]
    if not set(fetch_subject_list(data_dir=
                                  data_dir)).issuperset(set(subjects)):
        raise ValueError('Wrong subjects.')

    res = []
    if output == 'fsl':
        for subject in subjects:
            subject_dir = join(data_dir, str(subject), 'MNINonLinear',
                               'Results')
            for i, contrast in enumerate(CONTRASTS):
                task_name = contrast[0]
                contrast_idx = contrast[1]
                contrast_name = contrast[2]
                if level == 2:
                    z_map = join(subject_dir, "tfMRI_%s/tfMRI_%s_hp200_s4_"
                                              "level2vol.feat/cope%i.feat/"
                                              "stats/zstat1.nii.gz"
                                 % (task_name, task_name, contrast_idx))
                    if os.path.exists(z_map) or not on_disk:
                        res.append({'z_map': z_map,
                                    'subject': subject,
                                    'task': task_name,
                                    'contrast': contrast_name,
                                    'direction': 'level2'
                                    })
                    else:
                        break
                else:
                    raise ValueError("Can only output level 2 images"
                                     "with output='fsl'")
    else:
        source_dir = join(data_dir, 'glm')
        if level == 2:
            directions = ['level2']
        elif level == 1:
            directions = ['LR', 'RL']
        else:
            raise ValueError('Level should be 1 or 2, got %s' % level)
        for subject in subjects:
            subject_dir = join(source_dir, str(subject))
            for contrast in CONTRASTS:
                task_name = contrast[0]
                contrast_name = contrast[2]
                for direction in directions:
                    z_dir = join(subject_dir, task_name, direction,
                                 'z_maps')
                    effect_dir = join(subject_dir, task_name, direction,
                                      'effects_maps')
                    z_map = join(z_dir, 'z_' + contrast_name +
                                 '.nii.gz')
                    effect_map = join(effect_dir, 'effects_' + contrast_name +
                                      '.nii.gz')
                    if ((os.path.exists(z_map) and os.path.exists(effect_map))
                        or not on_disk):
                        res.append({'z_map': z_map,
                                    'effect_map': effect_map,
                                    'subject': subject,
                                    'task': task_name,
                                    'contrast': contrast_name,
                                    'direction': direction
                                    })
    res = pd.DataFrame(res)
    if not res.empty:
        res.set_index(['subject', 'task', 'contrast', 'direction'],
                      inplace=True)
        res.sort_index(ascending=True, inplace=True)
    return res


def fetch_behavioral_data(data_dir=None,
                          restricted=False,
                          overwrite=False):
    _, _, username, password = get_credentials(data_dir=data_dir)
    data_dir = get_data_dirs(data_dir)[0]
    behavioral_dir = join(data_dir, 'behavioral')
    if not os.path.exists(behavioral_dir):
        os.makedirs(behavioral_dir)
    csv_unrestricted = join(behavioral_dir, 'hcp_unrestricted_data.csv')
    if not os.path.exists(csv_unrestricted) or overwrite:
        result = _fetch_file(data_dir=data_dir,
                             url='https://db.humanconnectome.org/REST/'
                                 'search/dict/Subject%20Information/results?'
                                 'format=csv&removeDelimitersFromFieldValues'
                                 '=true'
                                 '&restricted=0&project=HCP_900',
                             username=username, password=password)
        os.rename(result, csv_unrestricted)
    csv_restricted = join(behavioral_dir, 'hcp_restricted_data.csv')
    df_unrestricted = pd.read_csv(csv_unrestricted)
    df_unrestricted.set_index('Subject', inplace=True)
    if restricted and not os.path.exists(csv_restricted):
        warnings.warn("Cannot automatically retrieve restricted data. "
                      "Please create the file '%s' manually" %
                      csv_restricted)
        restricted = False
    if not restricted:
        df = df_unrestricted
    else:
        df_restricted = pd.read_csv(csv_restricted)
        df_restricted.set_index('Subject', inplace=True)
        df = df_unrestricted.join(df_restricted, how='outer')
    df.sort_index(ascending=True, inplace=True)
    df.index.names = ['subject']
    return df


def fetch_subject_list(data_dir=None, n_subjects=None, only_terminated=True):
    df = fetch_behavioral_data(data_dir=data_dir)
    if only_terminated:
        indices = np.logical_and(df['3T_RS-fMRI_PctCompl'] == 100,
                                 df['3T_tMRI_PctCompl'] == 100)
        df = df.loc[indices]
    return df.iloc[:n_subjects].index. \
        get_level_values('subject').unique().tolist()


def download_experiment(subject,
                        data_dir=None,
                        data_type='rest',
                        tasks=None,
                        sessions=None,
                        overwrite=False,
                        mock=False,
                        verbose=0):
    aws_key, aws_secret, _, _ = get_credentials(data_dir)
    bucket = _init_s3_connection(aws_key, aws_secret, 'hcp-openaccess')
    targets = fetch_hcp_timeseries(data_dir=data_dir,
                                   subjects=subject,
                                   data_type=data_type,
                                   tasks=tasks,
                                   on_disk=False,
                                   sessions=sessions).values.ravel().tolist()
    keys = list(map(_convert_to_s3_target, targets))

    try:
        download_from_s3(bucket, keys[0], targets[0], mock=True,
                         verbose=0)
    except FileNotFoundError:
        return

    if verbose > 0:
        if data_type == 'task':
            print('Downloading files for subject %s,'
                  ' tasks %s' % (subject, tasks))
        else:
            print('Downloading files for subject %s,'
                  ' session %s' % (subject, sessions))
    for key, target in zip(keys, targets):
        dirname = os.path.dirname(target)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        try:
            download_from_s3(bucket, key, target, mock=mock,
                             overwrite=overwrite, verbose=verbose - 1)
        except FileNotFoundError:
            pass
        except ConnectionError:
            os.unlink(target)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            msg = '\n'.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback))
            target += '-error'
            with open(target, 'w+') as f:
                f.write(msg)


def download_from_s3(bucket, key, target, mock=False,
                     overwrite=False, verbose=0):
    """Download file from bucket
    """
    my_key = Key(bucket)
    my_key.key = key
    if my_key.exists():
        s3fid = bucket.get_key(key)
        if not mock:
            if not os.path.exists(target) or overwrite:
                if verbose:
                    print('Downloading %s from %s' % (target, key))
                s3fid.get_contents_to_filename(target)
                name, ext = os.path.splitext(target)
                if ext == '.gz':
                    try:
                        _ = nibabel.load(target).get_shape()
                        if verbose:
                            print('Nifti consistency checked.')
                    except:
                        raise ConnectionError('Corrupted download')
            else:
                if verbose:
                    print('Skipping %s as it already exists' % target)
        else:
            if verbose:
                print('Mock download %s from %s' % (target, key))
    else:
        raise FileNotFoundError('File does not exist on S3')


def get_data_dirs(data_dir=None):
    """ Returns the directories in which modl looks for data.

    This is typically useful for the end-user to check where the data is
    downloaded and stored.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    paths: list of strings
        Paths of the dataset directories.

    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :
    1. the keyword argument data_dir
    2. the global environment variable MODL_SHARED_DATA
    3. the user environment variable MODL_DATA
    4. modl_data in the user home folder
    """

    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend(data_dir.split(os.pathsep))

    # If data_dir has not been specified, then we crawl default locations
    if data_dir is None:
        global_data = os.getenv('HCP_SHARED_DATA')
        if global_data is not None:
            paths.extend(global_data.split(os.pathsep))

        local_data = os.getenv('HCP_DATA')
        if local_data is not None:
            paths.extend(local_data.split(os.pathsep))

        paths.append(os.path.expanduser('~/HCP900').split(os.pathsep))
    return paths


def get_credentials(filename=None, data_dir=None):
    """Retrieve credentials for COnnectomeDB and S3 bucket access.

    First try to look whether

    Parameters
    ----------
    filename: str,
        Filename of
    """
    try:
        if filename is None:
            filename = 'credentials.txt'
        if not os.path.exists(filename):
            data_dir = get_data_dirs(data_dir)[0]
            filename = join(data_dir, filename)
            if not os.path.exists(filename):
                if ('HCP_AWS_KEY' in os.environ
                    and 'HCP_AWS_SECRET_KEY' in os.environ
                    and 'CDB_USERNAME' in os.environ
                    and 'CDB_PASSWORD' in os.environ):
                    aws_key = os.environ['HCP_AWS_KEY']
                    aws_secret = os.environ['HCP_AWS_SECRET_KEY']
                    cdb_username = os.environ['CDB_USERNAME']
                    cdb_password = os.environ['CDB_PASSWORD']
                    return aws_key, aws_secret, cdb_username, cdb_password
                else:
                    raise KeyError('Could not find environment variables.')
        file = open(filename, 'r')
        return file.readline()[:-1].split(',')
    except (KeyError, FileNotFoundError):
        raise ValueError("Cannot find credentials. Provide them"
                         "in a file credentials.txt where the script is "
                         "executed, or in the HCP directory, or in"
                         "environment variables.")


def fetch_hcp_mask(data_dir=None, url=None, resume=True):
    data_dir = get_data_dirs(data_dir)[0]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_dir = join(data_dir, 'parietal')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if url is None:
        url = 'http://amensch.fr/data/cogspaces/mask/mask_img.nii.gz'
    _fetch_file(url, data_dir, resume=resume)
    return join(data_dir, 'mask_img.nii.gz')


def fetch_hcp(data_dir=None, n_subjects=None, subjects=None,
              from_file=False,
              on_disk=True):
    root = get_data_dirs(data_dir)[0]
    mask = fetch_hcp_mask(data_dir)
    if not from_file:
        rest = fetch_hcp_timeseries(data_dir, data_type='rest',
                                    n_subjects=n_subjects, subjects=subjects,
                                    on_disk=on_disk)
        task = fetch_hcp_timeseries(data_dir, data_type='task',
                                    n_subjects=n_subjects, subjects=subjects,
                                    on_disk=on_disk)
        contrasts = fetch_hcp_contrasts(data_dir,
                                        output='nistats',
                                        n_subjects=n_subjects,
                                        subjects=subjects,
                                        on_disk=on_disk)
        behavioral = fetch_behavioral_data(data_dir)
        indices = []
        for df in rest, task, contrasts:
            if not df.empty:
                indices.append(df.index.get_level_values('subject').
                               unique().values)
        if indices:
            index = indices[0]
            for this_index in indices[1:]:
                index = np.union1d(index, this_index)
            behavioral = behavioral.loc[index]
        else:
            behavioral = pd.DataFrame([])
    else:
        rest = pd.read_csv(join(root, 'parietal', 'rest.csv'))
        task = pd.read_csv(join(root, 'parietal', 'task.csv'))
        contrasts = pd.read_csv(join(root, 'parietal',
                                     'contrasts.csv'))
        behavioral = pd.read_csv(join(root, 'parietal',
                                      'behavioral.csv'))
        behavioral.set_index('subject', inplace=True)
        rest.set_index(['subject', 'session', 'direction'], inplace=True)
        task.set_index(['subject', 'task', 'direction'], inplace=True)
        contrasts.set_index(['subject', 'task', 'contrast', 'direction'],
                            inplace=True)
        if subjects is None:
            subjects = fetch_subject_list(data_dir=data_dir,
                                          n_subjects=n_subjects)
        rest = rest.loc[subjects]
        task = task.loc[subjects]
        contrasts = contrasts.loc[subjects]
        behavioral = behavioral.loc[subjects]

    return Bunch(rest=rest,
                 contrasts=contrasts,
                 task=task,
                 behavioral=behavioral,
                 mask=mask,
                 root=root)


def dump_hcp_csv(data_dir=None):
    dataset = fetch_hcp(data_dir, on_disk=True)
    data_dir = get_data_dirs(data_dir)[0]
    dataset.rest.to_csv(join(data_dir, 'parietal',
                             'rest.csv'))
    dataset.task.to_csv(join(data_dir, 'parietal',
                             'task.csv'))
    dataset.contrasts.to_csv(join(data_dir, 'parietal',
                                  'contrasts.csv'))
    dataset.behavioral.to_csv(join(data_dir, 'parietal',
                                   'behavioral.csv'))
