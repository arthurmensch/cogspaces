import json
from os.path import join

import numpy as np
import torch
from joblib import dump, load
from nilearn.input_data import NiftiMasker

from cogspaces.datasets import fetch_atlas_modl, fetch_mask, \
    load_reduced_loadings


def save(target_encoder, standard_scaler, estimator, metrics, info, config, output_dir):
    dump(target_encoder, join(output_dir, 'target_encoder.pkl'))
    dump(standard_scaler, join(output_dir, 'standard_scaler.pkl'))
    dump(estimator, join(output_dir, 'estimator.pkl'))
    dump(metrics, join(output_dir, 'metrics.pkl'))
    with open(join(output_dir, 'info.json'), 'w+') as f:
        json.dump(info, f)
    with open(join(output_dir, 'config.json'), 'w+') as f:
        json.dump(config, f)


def get_components(output_dir, dl=False, return_type='img'):
    if dl:
        estimator = load(join(output_dir, 'estimator.pkl'))
        components = estimator.components_dl_
    else:
        module = get_module(output_dir)
        components = module.embedder.linear.weight.detach().numpy()

    if return_type in ['img', 'arrays_full']:
        modl_atlas = fetch_atlas_modl()
        mask = fetch_mask()
        dictionary = modl_atlas['components_453_gm']
        masker = NiftiMasker(mask_img=mask).fit()
        components = masker.transform(dictionary)
        components_full = components.dot(dictionary)
        if return_type == 'img':
            components_img = masker.inverse_transform(components_full)
            return components_img
        elif return_type == 'arrays_full':
            components_full = components.dot(dictionary)
            return components_full
    elif return_type == 'arrays':
        return components


def get_names(output_dir):
    target_encoder = load(join(output_dir, 'target_encoder.pkl'))
    names = {study: le['contrast'].classes_.tolist()
             for study, le in target_encoder.le_.items()}
    full_names = ['%s::%s' % (study, contrast)
                  for study, contrasts in names.items()
                  for contrast in contrasts]
    return names, full_names


def get_classifs(output_dir, return_type='img'):
    module = get_module(output_dir)

    module.eval()

    studies = module.classifiers.keys()
    in_features = module.embedder.linear.in_features

    with torch.no_grad():
        classifs = module({study: torch.eye(in_features)
                           for study in studies}, logits=True)
        biases = module({study: torch.zeros((1, in_features))
                         for study in studies}, logits=True)
        classifs = {study: classifs[study] - biases[study]
                    for study in studies}
        classifs = {study: classif - classif.mean(dim=0, keepdim=True)
                    for study, classif in classifs.items()}
    classifs = {study: classif.numpy().T
                for study, classif in classifs.items()}
    if return_type in ['img', 'arrays_full']:
        modl_atlas = fetch_atlas_modl()
        mask = fetch_mask()
        dictionary = modl_atlas['components_453_gm']
        masker = NiftiMasker(mask_img=mask).fit()
        dictionary = masker.transform(dictionary)
        classifs_full = {study: classif.dot(dictionary)
                         for study, classif in classifs.items()}
        if return_type == 'img':
            classifs_img = masker.inverse_transform(
                np.concatenate(list(classifs_full.values()), axis=0))
            return classifs_img
        else:
            return classifs_full
    elif return_type == 'arrays':
        return classifs


def get_module(output_dir):
    estimator = load(join(output_dir, 'estimator.pkl'))
    if hasattr(estimator, 'classifier_'):
        module = estimator.classifier_.module_
    else:
        module = estimator.module_
    revert = torch.sum(module.embedder.linear.weight.detach(), dim=1) < 0
    module.embedder.linear.weight.data[revert] *= - 1
    module.embedder.linear.bias.data[revert] *= - 1
    for study, classifier in module.classifiers.items():
        classifier.batch_norm.running_mean[revert] *= -1
        classifier.linear.weight.data[:, revert] *= -1

    return module


def compute_grades(output_dir, grade_type='data_z_score'):
    grades = {}
    if grade_type == 'data_z_score':
        module = get_module(output_dir)
        target_encoder = load(join(output_dir, 'target_encoder.pkl'))
        data, target = load_reduced_loadings()
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
    elif grade_type == 'loadings':
        module = get_module(output_dir)
        for study, classifier in module.classifiers.items():
            in_features = classifier.linear.in_features
            classifier.eval()
            with torch.no_grad():
                loadings = (classifier(torch.eye(in_features),
                                       logits=True) - classifier(
                    torch.zeros(1, in_features),
                    logits=True)).transpose(0, 1).numpy()
                loadings -= loadings.mean(axis=0, keepdims=True)
                grades[study] = loadings

    elif grade_type == 'cosine_similarities':
        classifs_full = get_classifs(output_dir,
                                     return_type='arrays_full')
        components_full = get_components(output_dir,
                                         return_type='arrays_full')
        threshold = np.percentile(np.abs(components_full),
                                  100. * (1 - 1. / len(components_full)))
        components_full[components_full < threshold] = 0
        for study, classif_full in classifs_full.items():
            classif_full -= classif_full.mean(axis=0, keepdims=True)
            grades[study] = (
                    classif_full.dot(components_full.T)
                    / np.sqrt(np.sum(classif_full ** 2, axis=1)[:, None])
                    / np.sqrt(np.sum(components_full ** 2, axis=1)[None, :])
            )
    else:
        raise ValueError
    full_grades = np.concatenate(list(grades.values()), axis=0)

    names, full_names = get_names(output_dir)
    sorted_grades = []
    sorted_full_grades = []
    for i in range(full_grades.shape[1]):
        these_sorted_grades = {}
        for study, these_grades in grades.items():
            sort = np.argsort(these_grades[:, i])[::-1]
            these_sorted_grades[study] = {contrast: float(grade) for
                                          contrast, grade
                                          in zip(np.array(names[study])[sort],
                                                 these_grades[:, i][sort])}
        sorted_grades.append(these_sorted_grades)
        sort = np.argsort(full_grades[:, i])[::-1]
        sorted_full_grades.append({contrast: float(grade) for contrast, grade
                                   in zip(np.array(full_names)[sort],
                                          full_grades[:, i][sort])})
    grades = {'study': sorted_grades,
              'full': sorted_full_grades}
    return grades


def components_html(output_dir, components_dir):
    with open('plot_maps.html', 'r') as f:
        template = f.read()
    template = Template(template)
    imgs = []
    for i in range(128):
        title = 'components_%i' % i
        view_types = ['stat_map', 'glass_brain',
                      ]
        srcs = []
        for view_type in view_types:
            src = join(components_dir, '%s_%s.png' % (title, view_type))
            srcs.append(src)
        for grade_type in ['cosine_similarities', 'loadings']:
            wc_dir = 'wc_%s' % grade_type
            srcs.append(join(wc_dir, 'wc_single_%i.png' % i))
            srcs.append(join(wc_dir, 'wc_cat_%i.png' % i))
        imgs.append((srcs, title))
    html = template.render(imgs=imgs)
    output_file = join(output_dir, 'components.html')
    with open(output_file, 'w+') as f:
        f.write(html)


def classifs_html(output_dir, classifs_dir):
    from jinja2 import Template
    with open('plot_maps.html', 'r') as f:
        template = f.read()
    names, full_names = get_names(output_dir)
    template = Template(template)
    imgs = []
    for name in full_names:
        view_types = ['stat_map', 'glass_brain',
                      ]
        srcs = []
        for view_type in view_types:
            src = join(classifs_dir, '%s_%s.png' % (name, view_type))
            srcs.append(src)
        imgs.append((srcs, name))
    html = template.render(imgs=imgs)
    output_file = join(output_dir, 'classifs.html')
    with open(output_file, 'w+') as f:
        f.write(html)


def compute_nifti(output_dir):
    components_imgs = get_components(output_dir)
    components_imgs.to_filename(join(output_dir, 'components.nii.gz'))
    classifs_imgs = get_classifs(output_dir)
    classifs_imgs.to_filename(join(output_dir, 'classifs.nii.gz'))
    return classifs_imgs, components_imgs