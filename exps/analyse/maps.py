from itertools import islice

import json
import numpy as np
import os
import re
import torch
from joblib import load, delayed, Parallel, Memory
from matplotlib.testing.compare import get_cache_dir
from nilearn.input_data import NiftiMasker
from os.path import join

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir, get_data_dir
from cogspaces.plotting import plot_all
from exps.train import load_data

mem = Memory(cachedir=get_cache_dir())


def inspect_components(output_dir, dl=False):
    estimator = load(join(output_dir, 'estimator.pkl'))
    if dl:
        components = estimator.components_dl_
    else:
        components = estimator.components_
    dictionary, masker = get_proj_and_masker()
    components_full = components.dot(dictionary)
    components_img = masker.inverse_transform(components_full)
    filename = join(output_dir, '%s.nii.gz' % ('components_dl'
                                               if dl else 'components'))
    components_img.to_filename(filename)
    return masker, dictionary, components, components_full


def inspect_classification(output_dir):
    (masker, dictionary, components,
     components_full) = mem.cache(inspect_components)(output_dir, dl=False)
    estimator = load(join(output_dir, 'estimator.pkl'))
    target_encoder = load(join(output_dir, 'target_encoder.pkl'))
    module = estimator.classifier_.module_

    names = {study: le['contrast'].classes_.tolist()
             for study, le in target_encoder.le_.items()}

    classifs = {}
    classifs_full = {}
    for study, classifier in module.classifiers.items():
        these_classif = classifier.linear.weight.detach()
        multiplier = (classifier.batch_norm.weight.detach()
                      / torch.sqrt(classifier.batch_norm.running_var))
        these_classif *= multiplier
        classifs[study] = these_classif.numpy()
        classifs_full[study] = classifs[study].dot(components_full)
    classifs_img = masker.inverse_transform(
        np.concatenate(list(classifs_full.values()), axis=0))
    classifs_img.to_filename(join(output_dir, 'classifs.nii.gz'))
    with open(join(output_dir, 'names.json'), 'w+') as f:
        json.dump(names, f)
    return (masker, dictionary, components, components_full, classifs,
            classifs_full, names)


def grade_components(output_dir, grade_type='data_z_score'):
    (masker, dictionary, components, components_full, classifs,
     classifs_full, names) = mem.cache(inspect_classification)(output_dir)

    estimator = load(join(output_dir, 'estimator.pkl'))

    grades = {}
    if grade_type == 'data_z_score':
        module = estimator.classifier_.module_
        target_encoder = load(join(output_dir, 'target_encoder.pkl'))
        data, target = load_data(join(get_data_dir(), 'reduced_512'), 'all')
        data = {study: torch.from_numpy(this_data).float() for study, this_data
                in
                data.items()}
        target = target_encoder.transform(target)
        target = {study: torch.from_numpy(this_target['contrast'].values)
                  for study, this_target in target.items()}

        for study, this_data in data.items():
            this_target = target[study]
            n_contrasts = len(target_encoder.le_[study]['contrast'].classes_)
            preds = []
            z_scores = []
            for contrast in range(n_contrasts):
                data_contrast = this_data[this_target == contrast]
                with torch.no_grad():
                    module.eval()
                    pred = module.embedder(data_contrast)
                    z_scores.append(pred.mean(dim=0, keepdim=True))
                preds.append(pred)
            z_scores = torch.cat(z_scores, dim=0)
            preds = torch.cat(preds, dim=0)
            preds -= torch.mean(preds, dim=0, keepdim=True)
            std = torch.sqrt(torch.sum(preds ** 2, dim=0) /
                             (preds.shape[0] - 1))
            z_scores /= std[None, :]
            grades[study] = z_scores.numpy()

    elif grade_type == 'cosine_similarities':
        for study, classif in classifs.items():
            grades[study] = (classif_full.dot(components_full.T) /
                             np.sqrt(
                                 np.sum(classif_full ** 2, axis=1)[:, None])
                             / np.sqrt(
                        np.sum(components_full ** 2, axis=1)[None, :]))
    elif grade_type == 'log_odd':
        for study, classif in classifs.items():
            grades[study] = np.exp(classif)

    n_components = components_full.shape[0]

    full_grades = np.concatenate(list(grades.values()), axis=0)
    names = {study: np.array(these_names) for study, these_names
             in names.items()}
    full_names = np.array(['%s::%s' % (study, contrast)
                           for study, contrasts in names.items()
                           for contrast in contrasts])
    sorted_grades = []
    sorted_full_grades = []
    for i in range(n_components):
        these_sorted_grades = {}
        for study, these_grades in grades.items():
            sort = np.argsort(these_grades[:, i])[::-1]
            these_sorted_grades[study] = {contrast: float(grade) for
                                          contrast, grade
                                          in zip(names[study][sort],
                                                 these_grades[:, i][sort])}
        sorted_grades.append(these_sorted_grades)
        sort = np.argsort(full_grades[:, i])[::-1]
        sorted_full_grades.append({contrast: float(grade) for contrast, grade
                                   in zip(full_names[sort],
                                          full_grades[:, i][sort])})
    grades = {'grades': sorted_grades,
              'full_grades': sorted_full_grades}
    with open(join(output_dir, 'grades_%s.json' % grade_type), 'w+') as f:
        json.dump(grades, f)

    return grades


def get_proj_and_masker():
    modl_atlas = fetch_atlas_modl()
    dictionary = modl_atlas['components512']
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    dictionary = masker.transform(dictionary)
    return dictionary, masker


def inspect_all(output_dir, n_jobs=1):
    regex = re.compile(r'[0-9]+$')
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(inspect_components)(join(output_dir, this_dir), n_jobs=1)
        for this_dir in filter(regex.match, os.listdir(output_dir)))


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'single_full')
    inspect_components(output_dir)
    # inspect_classification(output_dir)

    # with open(join(output_dir, 'names.json'), 'r') as f:
    #     names = json.load(f)
    # full_names = np.array(['%s::%s' % (study, contrast)
    #                        for study, contrasts in names.items()
    #                        for contrast in contrasts])
    # plot_all(join(output_dir, 'classifs.nii.gz'),
    #          output_dir=join(output_dir, 'classif'),
    #          name='classif',
    #          names=full_names,
    #          n_jobs=3)

    for grade_type in ['cosine_similarities']:
        grade_components(output_dir, grade_type=grade_type)

        with open(join(output_dir, 'grades_%s.json' % grade_type), 'r') as f:
            grades = json.load(f)
        # texts = ["""<ul>\n""" + """\n""".join(
        #     """<li>%s : %.3f</li>""" % (contrast, these_grades[contrast])
        #     for contrast in islice(filter(lambda x: 'effects_of_interest' not in x,
        #                                   these_grades), 0, 10))
        #          + """</ul>""" for these_grades in grades['full_grades']]

        texts = ["""<ul>\n""" + """\n""".join(
            """<li>%s : %s, %.3f</li>""" % (study, contrast, study_graves[contrast])
            for study, study_graves in these_grades.items()
            for contrast in islice(study_graves, 0, 1))
                 + """</ul>""" for these_grades in grades['grades']]

        plot_all(join(output_dir, 'components.nii.gz'),
                 output_dir=join(output_dir, 'components'),
                 name='component',
                 filename=grade_type + '_study',
                 draw=False,
                 texts=texts, n_jobs=3)
