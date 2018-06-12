from itertools import islice
from wordcloud import WordCloud

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


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)



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
            classif -= classif.mean(axis=0, keepdims=True)
            classif = classif.dot(components)
            grades[study] = (classif.dot(components.T)
                             / np.sqrt(np.sum(classif ** 2, axis=1)[:, None])
                             / np.sqrt(np.sum(components ** 2, axis=1)[None, :])
                             )
            # grades[study] -= grades[study].mean(axis=0, keepdims=True)
    elif grade_type == 'log_odd':
        for study, classif in classifs.items():
            classif -= classif.mean(axis=0, keepdims=True)
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


def rgb2hex(r,g,b):
    return f'#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}'


def plot_word_clouds(output_dir):
    import seaborn as sns
    import matplotlib.pyplot as plt

    wc_dir = join(output_dir, 'components', 'wc')

    if not os.path.exists(wc_dir):
        os.makedirs(wc_dir)

    with open(join(output_dir, 'grades_%s.json' % 'cosine_similarities'), 'r') as f:
        grades = json.load(f)
    with open(join(output_dir, 'names.json'), 'r') as f:
        names = json.load(f)
    for i, these_grades in enumerate(grades['full_grades']):
        contrasts = list(filter(
            lambda x: 'effects_of_interest' not in x, these_grades))[:15]
        frequencies = []
        studies = []
        for contrast in contrasts:
            grade = these_grades[contrast]
            study, contrast = contrast.split('::')
            if study == 'hcp':
                contrast = contrast.replace('LF', 'left foot')
                contrast = contrast.replace('RF', 'right foot')
                contrast = contrast.replace('LH', 'left hand')
                contrast = contrast.replace('RH', 'right hand')
            contrast = contrast.replace('clicGaudio', 'left audio click')
            contrast = contrast.replace('clicDaudio', 'right audio click')
            contrast = contrast.replace('calculvideo', 'video calculation')
            contrast = contrast.replace('calculaudio', 'audio calculation')

            terms = contrast.split('_')
            contrast = []
            for term in terms:
                if term == 'baseline':
                    break
                if term == 'vs':
                    break
                else:
                    contrast.append(term)
            if contrast:
                contrast = ' '.join(contrast[:3])
                curated = contrast.lower()
                frequencies.append((curated, grade))
                studies.append(study)
        print(frequencies, studies)
        colors = sns.color_palette('husl', 35)
        color_to_words = {rgb2hex(*color): [study] for color, study in zip(colors, names)}
        color_func = SimpleGroupedColorFunc(color_to_words, default_color='#ffffff')
        wc = WordCloud(color_func=color_func)
        wc.generate_from_frequencies(frequencies=frequencies,
                                     as_tuples=True,
                                     group_colors=studies)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.savefig(join(wc_dir, 'wc_%i.png' % i))
        plt.close(fig)


if __name__ == '__main__':
    output_dir = join(get_output_dir(), 'single_full')
    # inspect_components(output_dir)
    # inspect_components(output_dir, dl=True)
    # inspect_classification(output_dir)
    #
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
    # plot_all(join(output_dir, 'components_dl.nii.gz'),
    #          output_dir=join(output_dir, 'components_dl'),
    #          name='component_dl',
    #          draw=True,
    #          n_jobs=3)
    #
    grade_components(output_dir, grade_type='cosine_similarities')

    plot_word_clouds(output_dir)

    draw = False
    for grade_type in ['cosine_similarities']:

        with open(join(output_dir, 'grades_%s.json' % grade_type), 'r') as f:
            grades = json.load(f)

        for per_study in [False]:
            if per_study:
                texts = ["""<ul>\n""" + """\n""".join(
                    """<li>%s : %s, %.3f</li>""" % (
                    study, contrast, study_graves[contrast])
                    for study, study_graves in these_grades.items()
                    for contrast in islice(study_graves, 0, 1))
                         + """</ul>""" for these_grades in grades['grades']]
                filename = grade_type + '_study'
            else:
                texts = ["""<ul>\n""" + """\n""".join(
                    """<li>%s : %.3f</li>""" % (contrast, these_grades[contrast])
                    for contrast in islice(filter(lambda x: 'effects_of_interest' not in x,
                                                  these_grades), 0, 10))
                         + """</ul>""" for these_grades in grades['full_grades']]
                filename = grade_type
            plot_all(join(output_dir, 'components.nii.gz'),
                     output_dir=join(output_dir, 'components'),
                     name='component',
                     filename=filename,
                     draw=draw,
                     word_clouds=True,
                     texts=texts, n_jobs=3)
            draw = False
