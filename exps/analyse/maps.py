import numpy as np
import os
import re
from joblib import load, delayed, Parallel
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
    plot_all(filename, output_dir=join(output_dir, 'components'),
             name='components', n_jobs=n_jobs)


def inspect_classification(output_dir, n_jobs=3):
    estimator = load(join(output_dir, 'estimator.pkl'))
    module = estimator.module_
    components = module.embedder.linear.weight.detach().numpy()
    dictionary, masker = get_proj_and_masker()
    dictionary = components.dot(dictionary)

    classification = []
    for study, classifier in module.classifiers.items():
        classifier_coef = classifier.linear.weight.data.numpy()
        std = np.sqrt(classifier.batch_norm.running_var.data.numpy())
        classifier_coef /= std
        classification.append(classifier_coef)
    classification = np.concatenate(classification, axis=0)
    classification = classification.dot(dictionary)
    classification = masker.inverse_transform(classification)
    filename = join(output_dir, 'classification.nii.gz')
    classification.to_filename(filename)
    plot_all(filename, name='classification',
             output_dir=join(output_dir, 'components'),
             n_jobs=n_jobs)


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
    # inspect_all(join(get_output_dir(), 'factored'), n_jobs=10)
    inspect_components(join(get_output_dir(), 'factored', '1'), n_jobs=3)
    # inspect_components(join(get_output_dir(), 'multi_studies'))