import numpy as np
import os
import re
import torch
from jinja2 import Template
from joblib import load, Memory, dump
from matplotlib.colors import hsv_to_rgb
from matplotlib.testing.compare import get_cache_dir
from nilearn.datasets import fetch_surf_fsaverage5
from os.path import join
from sklearn.utils import check_random_state

from cogspaces.datasets.utils import get_output_dir, get_data_dir
from cogspaces.plotting import plot_word_clouds, plot_all
from cogspaces.utils import get_dictionary, get_masker
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
    revert = torch.sum(module.embedder.linear.weight.detach(), dim=1) < 0
    module.embedder.linear.weight.data[revert] *= - 1
    module.embedder.linear.bias.data[revert] *= - 1
    for study, classifier in module.classifiers.items():
        classifier.batch_norm.running_mean[revert] *= -1
        classifier.linear.weight.data[:, revert] *= -1

    return module


def get_grades(output_dir, grade_type='data_z_score'):
    grades = {}
    if grade_type == 'data_z_score':
        module = get_module(output_dir)
        target_encoder = load(join(output_dir, 'target_encoder.pkl'))
        data, target = load_data(join(get_data_dir(), 'reduced_512_gm'), 'all')
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
                grades[study] = (classifier(torch.eye(in_features),
                                            logits=True)
                                 - classifier(torch.zeros(1, in_features),
                                              logits=True)).transpose(0,
                                                                      1).numpy()

    elif grade_type == 'cosine_similarities':
        classifs_full = get_classifs(output_dir,
                                     return_type='arrays_full')
        components_full = get_components(output_dir,
                                         return_type='arrays_full')
        # metrics = pd.read_pickle(
        #     join(get_output_dir(), 'factored_gm', 'metrics.pkl'))
        # f1s = metrics['f1'].groupby(['study', 'contrast']).mean()
        #
        # names, full_names = get_names(output_dir)
        #
        for study, classif_full in classifs_full.items():
            # this_f1 = f1s.loc[study]
            # this_f1 = this_f1.loc[names[study]]
            classif_full -= classif_full.mean(axis=0, keepdims=True)
            grades[study] = (
                    classif_full.dot(components_full.T)
                    / np.sqrt(np.sum(classif_full ** 2, axis=1)[:, None])
                    / np.sqrt(np.sum(components_full ** 2, axis=1)[None, :])
            )  # * this_f1[:, None]
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


def components_html(output_dir, components_dir, wc_dir):
    with open('plot_maps.html', 'r') as f:
        template = f.read()
    template = Template(template)
    imgs = []
    for i in range(128):
        title = 'components_%i' % i
        view_types = ['stat_map', 'glass_brain',
                      # 'surf_stat_map_lateral_left',
                      # 'surf_stat_map_medial_left',
                      # 'surf_stat_map_lateral_right',
                      # 'surf_stat_map_medial_right'
                      ]
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
                      # 'surf_stat_map_lateral_left',
                      # 'surf_stat_map_medial_left',
                      # 'surf_stat_map_lateral_right',
                      # 'surf_stat_map_medial_right'
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
    fetch_surf_fsaverage5()

    components_imgs = get_components(output_dir)
    components_imgs.to_filename(join(output_dir, 'components.nii.gz'))
    classifs_imgs = get_classifs(output_dir)
    classifs_imgs.to_filename(join(output_dir, 'classifs.nii.gz'))


def compute_grades(output_dir):
    grades = get_grades(output_dir, grade_type='cosine_similarities')
    dump(grades, join(output_dir, 'grades.pkl'))


def plot_grades(output_dir, n_jobs):
    colors = np.load(join(output_dir, 'colors.npy'))
    grades = load(join(output_dir, 'grades.pkl'))


    plot_word_clouds(join(output_dir, 'wc'), grades, n_jobs=n_jobs,
                     colors=colors)


def plot_2d(output_dir, n_jobs=40):
    view_types = ['stat_map', 'glass_brain',
                  # 'surf_stat_map_lateral_left',
                  # 'surf_stat_map_medial_left',
                  # 'surf_stat_map_lateral_right',
                  # 'surf_stat_map_medial_right'
                  ]

    names, full_names = get_names(output_dir)

    # plot_all(join(output_dir, 'classifs.nii.gz'),
    #          output_dir=join(output_dir, 'classifs'),
    #          names=full_names,
    #          view_types=view_types,
    #          n_jobs=n_jobs)

    colors = np.load(join(output_dir, 'colors.npy'))

    plot_all(join(output_dir, 'components.nii.gz'),
             output_dir=join(output_dir, 'components'),
             names='components',
             colors=colors,
             view_types=view_types,
             n_jobs=n_jobs)


def make_report(output_dir):
    components_html(output_dir, 'components', 'wc')
    classifs_html(output_dir, 'classifs')


if __name__ == '__main__':
    n_jobs = 40

    regex = re.compile(r'[0-9]+$')
    full_names = []

    output_dir = join(get_output_dir(), 'components')
    # output_dir = join(get_output_dir(), 'factored_refit_gm_normal_init_full_rest_positive_notune')
    #
    for dirpath, dirnames, filenames in os.walk(output_dir):
        for dirname in filter(lambda f: re.match(regex, f), dirnames):
            full_name = join(dirpath, dirname)
            full_names.append(full_name)
    #
    rng = check_random_state(0)
    hs = np.linspace(0, 1, 128, endpoint=False)
    rgbs = [list(hsv_to_rgb((h, 1, 1))) for h in hs]
    colors = np.array(rgbs)
    rng.shuffle(colors)

    for full_name in full_names:
        np.save(join(full_name, 'colors.npy'), colors)

    # Parallel(n_jobs=n_jobs, verbose=10)(delayed(compute_nifti)(full_name)
    #                                     for full_name in full_names)
    # Parallel(n_jobs=n_jobs, verbose=10)(delayed(plot_3d)(full_name)
    #                                     for full_name in full_names)
    # for full_name in full_names:
    #     plot_2d(full_name, n_jobs=n_jobs)
    # Parallel(n_jobs=n_jobs, verbose=10)(delayed(compute_grades)(full_name)
    #                                     for full_name in full_names)
    for full_name in full_names:
        plot_grades(full_name, n_jobs=n_jobs)
    #
    # for full_name in full_names:
    #     make_report(full_name)