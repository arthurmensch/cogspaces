import json
import numpy as np
import torch
import torch.nn.functional as F
from jinja2 import Template
from joblib import load, Memory
from matplotlib.testing.compare import get_cache_dir
from nilearn.input_data import NiftiMasker
from os.path import join

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir, get_data_dir
from cogspaces.plotting import plot_word_clouds, plot_all
from exps.train import load_data

mem = Memory(cachedir=get_cache_dir())


def get_components(output_dir, dl=False, return_type='img'):
    if dl:
        estimator = load(join(output_dir, 'estimator.pkl'))
        components = estimator.components_dl_
    else:
        module = get_module(output_dir)
        components = module.embedder.linear.weight.detach().numpy()

    if return_type in ['img', 'arrays_full']:
        dictionary = get_dictionary()
        components_full = components.dot(dictionary)
        if return_type == 'img':
            masker = get_masker()
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
    classifs = {study: classif.numpy().T
                for study, classif in classifs.items()}
    if return_type in ['img', 'arrays_full']:
        dictionary = get_dictionary()
        classifs_full = {study: classif.dot(dictionary)
                         for study, classif in classifs.items()}
        if return_type == 'img':
            masker = get_masker()
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
    return module


def get_grades(output_dir, grade_type='data_z_score'):
    grades = {}
    if grade_type == 'data_z_score':
        module = get_module(output_dir)
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
    else:
        classifs_full = get_classifs(output_dir,
                                     return_type='arrays_full')
        components_full = get_components(output_dir,
                                         return_type='arrays_full')
        if grade_type == 'cosine_similarities':
            for study, classif_full in classifs_full.items():
                classif_full -= classif_full.mean(axis=0, keepdims=True)
                grades[study] = (
                        classif_full.dot(components_full.T)
                        / np.sqrt(np.sum(classif_full ** 2, axis=1)[:, None])
                        / np.sqrt(
                    np.sum(components_full ** 2, axis=1)[None, :])
                )
        elif grade_type == 'log_odd':
            classifs = get_classifs(output_dir, return_type='arrays')
            for study, classif in classifs.items():
                classif -= classif.mean(axis=0, keepdims=True)
                grades[study] = np.exp(classif)
        elif grade_type == 'model':
            module = get_module(output_dir)
            components = get_components(output_dir,
                                        return_type='components')
            with torch.no_grad():
                logits = module({study: torch.from_numpy(components)
                                 for study in module.classifiers})
            grades = {study: F.softmax(logit, dim=1).transpose(0, 1).numpy()
                      for study, logit in logits.items()}

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


def get_masker():
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    return masker


def get_dictionary():
    modl_atlas = fetch_atlas_modl()
    dictionary = modl_atlas['components512_gm']
    masker = get_masker()
    dictionary = masker.transform(dictionary)
    return dictionary


def components_html(output_dir, components_dir, wc_dir):
    with open('plot_maps.html', 'r') as f:
        template = f.read()
    template = Template(template)
    imgs = []
    for i in range(128):
        title = 'components_%i' % i
        view_types = ['stat_map', 'glass_brain',
                      'surf_stat_map_lateral_left',
                      'surf_stat_map_medial_left',
                      'surf_stat_map_lateral_right',
                      'surf_stat_map_medial_right']
        srcs = []
        for view_type in view_types:
            src = join(components_dir, '%s_%s.png' % (title, view_type))
            srcs.append(src)
        srcs.append(join(wc_dir, 'wc_single_%i.png' % i))
        srcs.append(join(wc_dir, 'wc_cat_%i.png' % i))
        imgs.append((srcs, title))
    html = template.render(imgs=imgs)
    output_file = join(output_dir, 'components.html')
    with open(output_file, 'w+') as f:
        f.write(html)


def classifs_html(output_dir, classifs_dir):
    with open('plot_maps.html', 'r') as f:
        template = f.read()
    names, full_names = get_names(output_dir)
    template = Template(template)
    imgs = []
    for name in full_names:
        view_types = ['stat_map', 'glass_brain',
                      'surf_stat_map_lateral_left',
                      'surf_stat_map_medial_left',
                      'surf_stat_map_lateral_right',
                      'surf_stat_map_medial_right']
        srcs = []
        for view_type in view_types:
            src = join(classifs_dir, '%s_%s.png' % (name, view_type))
            srcs.append(src)
        imgs.append((srcs, name))
    html = template.render(imgs=imgs)
    output_file = join(output_dir, 'classifs.html')
    with open(output_file, 'w+') as f:
        f.write(html)


def make_report(output_dir, dl=False):
    view_types = ['stat_map', 'glass_brain',
                  'surf_stat_map_lateral_left',
                  'surf_stat_map_medial_left',
                  'surf_stat_map_lateral_right',
                  'surf_stat_map_medial_right']

    # components_imgs = get_components(output_dir)
    # components_imgs.to_filename(join(output_dir, 'components.nii.gz'))
    # classifs_imgs = get_classifs(output_dir)
    # classifs_imgs.to_filename(join(output_dir, 'classifs.nii.gz'))
    #
    # grades = get_grades(output_dir, grade_type='cosine_similarities')
    # with open(join(output_dir, 'grades.json'), 'w+') as f:
    #     json.dump(grades, f)
    #
    # names, full_names = get_names(output_dir)
    #
    # fetch_surf_fsaverage5()
    # plot_all(join(output_dir, 'classifs.nii.gz'),
    #          output_dir=join(output_dir, 'classifs'),
    #          names=full_names,
    #          view_types=view_types,
    #          n_jobs=40)
    plot_all(join(output_dir, 'components.nii.gz'),
             output_dir=join(output_dir, 'components'),
             names='components',
             view_types=view_types,
             n_jobs=40)
    #
    #
    # if dl:
    #     components_imgs_dl = get_components(output_dir, dl=True)
    #     components_imgs_dl.to_filename(join(output_dir, 'components_dl.nii.gz'))
    #     plot_all(join(output_dir, 'components_dl.nii.gz'),
    #              output_dir=join(output_dir, 'components_dl'),
    #              names='component_dl',
    #              view_types=view_types,
    #              n_jobs=40)

    with open(join(output_dir, 'grades.json'), 'r') as f:
        grades = json.load(f)
    plot_word_clouds(join(output_dir, 'wc'), grades, n_jobs=40)

    components_html(output_dir, 'components', 'wc')
    # classifs_html(output_dir, 'classifs')


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'factored_refit_gm_full_notune', '1',)
    make_report(output_dir)
    output_dir = join(get_output_dir(), 'factored_refit_gm_low_lr', '1')
    make_report(output_dir)