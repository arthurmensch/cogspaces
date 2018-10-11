"""
    Analyse a model trained on data provided by load_reduced_loadings().
"""

import numpy as np
import torch
from cogspaces.datasets import fetch_atlas_modl, fetch_mask
from nilearn.input_data import NiftiMasker


# Note that 'components_453_gm' is currently hard-coded there
# Note that this module only works without standard_scaling


def compute_components(estimator, config, return_type='img'):
    module = curate_module(estimator)
    components = module.embedder.linear.weight.detach().numpy()
    if config['data']['reduced']:
        mask = fetch_mask()
        masker = NiftiMasker(mask_img=mask).fit()
        modl_atlas = fetch_atlas_modl()
        dictionary = modl_atlas['components_453_gm']
        dictionary = masker.transform(dictionary)
        components = components.dot(dictionary)
    if return_type == 'img':
        components_img = masker.inverse_transform(components)
        return components_img
    elif return_type == 'arrays':
        return components


def compute_classifs(estimator, standard_scaler, config, return_type='img'):
    if config['model']['estimator'] in ['factored', 'ensemble']:
        module = curate_module(estimator)
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

    elif config['model']['estimator'] == 'logistic':
        classifs = estimator.coef_
    else:
        raise NotImplementedError

    if config['data']['reduced']:
        mask = fetch_mask()
        masker = NiftiMasker(mask_img=mask).fit()
        modl_atlas = fetch_atlas_modl()
        dictionary = modl_atlas['components_453_gm']
        dictionary = masker.transform(dictionary)
        classifs = {study: classif.dot(dictionary)
                    for study, classif in classifs.items()}
    if return_type == 'img':
        classifs_img = masker.inverse_transform(
            np.concatenate(list(classifs.values()), axis=0))
        return classifs_img
    else:
        return classifs


def curate_module(estimator):
    module = estimator.module_
    revert = torch.sum(module.embedder.linear.weight.detach(), dim=1) < 0
    module.embedder.linear.weight.data[revert] *= - 1
    module.embedder.linear.bias.data[revert] *= - 1
    for study, classifier in module.classifiers.items():
        classifier.batch_norm.running_mean[revert] *= -1
        classifier.linear.weight.data[:, revert] *= -1

    return module


def compute_grades(estimator, standard_scaler, target_encoder,
                   config, grade_type='cosine_similarities',
                   ):
    grades = {}
    if grade_type == 'loadings':
        module = curate_module(estimator)
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
        classifs = compute_classifs(estimator, standard_scaler, config,
                                    return_type='arrays')
        components = compute_components(estimator, standard_scaler, config,
                                        return_type='arrays')
        threshold = np.percentile(np.abs(components),
                                  100. * (1 - 1. / len(components)))
        components[components < threshold] = 0
        for study, classif in classifs.items():
            classif -= classif.mean(axis=0, keepdims=True)
            grades[study] = (
                    classif.dot(components.T)
                    / np.sqrt(np.sum(classifs ** 2, axis=1)[:, None])
                    / np.sqrt(np.sum(components ** 2, axis=1)[None, :])
            )
    else:
        raise ValueError
    full_grades = np.concatenate(list(grades.values()), axis=0)

    names, full_names = compute_names(target_encoder)
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


def compute_names(target_encoder):
    names = {study: le['contrast'].classes_.tolist()
             for study, le in target_encoder.le_.items()}
    full_names = ['%s::%s' % (study, contrast)
                  for study, contrasts in names.items()
                  for contrast in contrasts]
    return names, full_names


def compute_nifti(estimator, standard_scaler, config):
    classifs_imgs = compute_classifs(estimator, standard_scaler, config,
                                     return_type='img')
    if config['model']['estimator'] == 'factored':
        if not config['data']['reduced']:
            raise NotImplementedError
        components_imgs = compute_components(estimator, config, standard_scaler)
        return classifs_imgs, components_imgs
    else:
        return classifs_imgs
