import numpy as np
import os
import re
import torch
from joblib import load, delayed, Parallel, dump
from nilearn.input_data import NiftiMasker
from os.path import join

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir, get_data_dir
from cogspaces.plotting import plot_all
from exps.train import load_data


def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)


def inspect_components(output_dir, n_jobs=3):
    estimator = load(join(output_dir, 'estimator.pkl'))
    components = estimator.module_.embedder.linear.weight.detach().numpy()
    dictionary, masker = get_proj_and_masker()
    components = components.dot(dictionary)
    components = masker.inverse_transform(components)
    filename = join(output_dir, 'components.nii.gz')
    components.to_filename(filename)

    # plot_all(filename, output_dir=join(output_dir, 'components'),
    #          name='components', n_jobs=n_jobs, verbose=0)


def inspect_components_dl(output_dir, n_jobs=3):
    estimator = load(join(output_dir, 'estimator.pkl'))
    components = estimator.components_
    components_dl = estimator.components_dl_
    dictionary, masker = get_proj_and_masker()

    comp_to_plot = {'components': components,
                    'component_dl': components_dl}

    print(components)
    print(components_dl)

    for name, this_comp in comp_to_plot.items():
        this_comp = this_comp.dot(dictionary)
        this_comp = masker.inverse_transform(this_comp)
        filename = join(output_dir, '%s.nii.gz' % name)
        this_comp.to_filename(filename)
        plot_all(filename, output_dir=join(output_dir, name),
                 name=name, n_jobs=n_jobs, verbose=0)


def inspect_classification(output_dir, n_jobs=3):
    estimator = load(join(output_dir, 'estimator.pkl'))
    target_encoder = load(join(output_dir, 'target_encoder.pkl'))
    module = estimator.classifier_.module_

    data, target = load_data(join(get_data_dir(), 'reduced_512'), 'all')
    target = target_encoder.transform(target)

    data = {study: torch.from_numpy(this_data).float() for study, this_data in
            data.items()}
    target = {study: torch.from_numpy(this_target['contrast'].values)
              for study, this_target in target.items()}

    dictionary, masker = get_proj_and_masker()
    components = module.components_
    full_components = components.dot(dictionary)

    filename = join(output_dir, 'components.nii.gz')
    component_imgs = masker.inverse_transform(full_components)
    filename = join(output_dir, 'components.nii.gz')
    component_imgs.to_filename(filename)

    plot_all(filename, name='components',
             texts=['Components %i' % i for i in range(components.shape[0])],
             output_dir=join(output_dir, 'components'),
             draw=False,
             n_jobs=n_jobs)

    module.eval()

    names = []
    for study, this_data in data.items():
        contrasts = target_encoder.le_[study]['contrast'].classes_
        names.extend(['%s_%s' % (study, name) for name in contrasts])
    names = np.array(names)

    z_scores = []
    for study, this_data in data.items():
        this_target = target[study]
        contrasts = target_encoder.le_[study]['contrast'].classes_
        study_all = []
        study_mean = []
        for contrast in range(len(contrasts)):
            data_contrast = this_data[this_target == contrast]
            with torch.no_grad():
                pred = module.classifiers[study].batch_norm(module.embedder(data_contrast))
            study_mean.append(pred.mean(dim=0, keepdim=True))
            study_all.append(pred)
        study_mean = torch.cat(study_mean, dim=0)
        study_all = torch.cat(study_all, dim=0)
        study_all -= torch.mean(study_all, dim=0, keepdim=True)
        study_std = torch.sqrt(torch.sum(study_all ** 2, dim=0) / (study_all.shape[0] - 1))
        study_mean /= study_std[None, :]
        z_scores.append(study_mean)
    z_scores = torch.cat(z_scores, dim=0)
    z_scores = z_scores.numpy()

    n_components = z_scores.shape[1]
    comp_text = []
    z_scores_per_comp = []
    for i in range(n_components):
        sort = np.argsort(z_scores[:, i])[::-1]
        z_scores = {tag: activation for tag, activation
                in zip(names[sort], z_scores[:, i][sort])}

    classification = []
    for study, this_data in data.items():
        classifier = module.classifiers[study]
        these_classif = classifier.linear.weight.detach()
        multiplier = (classifier.batch_norm.weight.detach()
                      / torch.sqrt(classifier.batch_norm.running_var))
        these_classif *= multiplier
        classification.append(these_classif)

    classification = torch.cat(classification, dim=0)
    components = components.numpy()
    components = components.dot(dictionary)
    classification = classification.numpy()
    classification = classification.dot(components)

    gram = (classification.dot(components.T) /
            np.sqrt(np.sum(classification ** 2, axis=1)[:, None])
            / np.sqrt(np.sum(components ** 2, axis=1)[None, :]))
    dump(gram, join(output_dir, 'gram.pkl'))
    gram = load(join(output_dir, 'gram.pkl'))
    print(gram)

    n_components = gram.shape[1]
    comp_text = []
    for i in range(n_components):
        sort = np.argsort(gram[:, i])[::-1]
        sort = sort[:10]
        tags = {tag: activation for tag, activation
                in zip(names[sort], gram[:, i][sort])}
        comp_text.append(repr(tags))

    filename = join(output_dir, 'components.nii.gz')
    plot_all(filename, name='components',
             texts=comp_text,
             output_dir=join(output_dir, 'components'),
             draw=False,
             n_jobs=n_jobs)


    # classification = masker.inverse_transform(classification)
    # filename = join(output_dir, 'classification.nii.gz')
    # classification.to_filename(filename)

    # dump(names, 'names.pkl')
    # plot_all(filename, name='classification',
    #          names=names,
    #          output_dir=join(output_dir, 'classification'),
    #          n_jobs=n_jobs)


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
    # inspect_all(join(get_output_dir(), 'factored_sparsify'), n_jobs=10)
    # inspect_all(join(get_output_dir(), 'factored_refit_cautious'), n_jobs=10)
    # inspect_all(join(get_output_dir(), 'factored_sparsify_less'), n_jobs=10)
    # inspect_components_dl(join(get_output_dir(), 'single_full'), n_jobs=3)
    inspect_classification(join(get_output_dir(), 'single_full'), n_jobs=3)
    # inspect_components(join(get_output_dir(), 'multi_studies'))