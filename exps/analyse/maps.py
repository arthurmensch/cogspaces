import numpy as np
import os
import re
import torch
import torch.nn.functional as F
from joblib import load, delayed, Parallel, dump
from nilearn.input_data import NiftiMasker
from os.path import join

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir
from cogspaces.plotting import plot_all


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

    # components = module.embedder.linear.weight.detach().numpy()
    # dictionary, masker = get_proj_and_masker()
    # dictionary = components.dot(dictionary)

    classification = []
    preds = []
    names = []
    for study, classifier in module.classifiers.items():
        classifier_coef = classifier.linear.weight.detach()
        multiplier = (classifier.batch_norm.weight.detach()\
                      / torch.sqrt(classifier.batch_norm.running_var))
        classifier_coef *= multiplier

        classifier_coef = classifier_coef.numpy()

        latent_size = classifier_coef.shape[1]
        input = torch.eye(latent_size)
        classifier.eval()
        pred = F.softmax(classifier.linear(input), dim=1)
        print('---------------')
        print(study)
        print('---------------')
        print(np.array2string(pred.detach().numpy(), precision=3))
        classification.append(classifier_coef)
        preds.append(pred)
        these_names = np.array(['%s_%s' % (study, name) for name in
                       target_encoder.le_[study]['contrast'].classes_])
        names.append(these_names)

    classification = np.concatenate(classification, axis=0)
    names = np.concatenate(names)

    classifications = {'classification': classification,}

    dump(classifications, join(output_dir, 'classifications.pkl'))
    #
    # for i in range(classification_std.shape[1]):
    #     sort = np.argsort(classification_std[:, i])[::-1]
    #     print(classification_std[:, i][sort][:5])
    #     print(names[sort][:5])

    # for name, classif in classifications.items():
    #     classif = classif.dot(dictionary)
    #     classif = masker.inverse_transform(classif)
    #     filename = join(output_dir, '%s.nii.gz' % name)
    #     classif.to_filename(filename)
        # plot_all(filename, name=name,
        #          names=names,
        #          output_dir=join(output_dir, 'components'),
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