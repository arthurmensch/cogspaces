import numpy as np
from joblib import load
from nilearn.input_data import NiftiMasker
from os.path import join

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_output_dir
from cogspaces.plotting import plot_all


def inspect_components(output_dir, n_jobs=20):
    estimator = load(join(output_dir, 'estimator.pkl'))
    components = estimator.module_.embedder.linear.weight.detach().numpy()
    dictionary, masker = get_proj_and_masker()
    components = components.dot(dictionary)
    components = masker.inverse_transform(components)
    filename = join(output_dir, 'components.nii.gz')
    components.to_filename(filename)
    plot_all(filename, output_dir=join(output_dir, 'components'),
             name='components', n_jobs=n_jobs)


def inspect_classification(output_dir, n_jobs=20):
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


if __name__ == '__main__':
    inspect_components(join(get_output_dir(), 'init_refit_finetune',
                                '3'))