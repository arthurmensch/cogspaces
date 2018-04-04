import os
from os.path import join

import matplotlib.pyplot as plt
from joblib import load, dump
from nilearn.image import index_img
from nilearn.plotting import plot_stat_map
from scipy.linalg import svd

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import get_output_dir
from cogspaces.introspect.maps import maps_from_model


def plot_components(components, names, output_dir):
    for study in components:
        this_img = components[study]
        these_names = names[study]

        for i, name in enumerate(these_names):
            full_name = study + ' ' + name
            fig = plt.figure()
            plot_stat_map(index_img(this_img, i), figure=fig, title=full_name)
            plt.savefig(join(output_dir, '%s.png' % full_name))
            plt.close(fig)


def compute_components(output_dir, lstsq):
    estimator = load(join(output_dir, 'estimator.pkl'))
    target_encoder = load(join(output_dir, 'target_encoder.pkl'))
    standard_scaler = load(join(output_dir, 'standard_scaler.pkl'))

    modl_atlas = fetch_atlas_modl()
    dictionary = modl_atlas['components512']
    components, names = maps_from_model(estimator, dictionary,
                                        target_encoder,
                                        standard_scaler,
                                        lstsq=lstsq)
    plot_dir = join(output_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    dump(names, join(plot_dir, 'names.pkl'))
    for study, this_components in components.items():
        this_components.to_filename(join(plot_dir,
                                         'components_%s.nii.gz' % study))
        plot_components(components, names, plot_dir)


def plot_activation(output_dir):
    test_latents = load(join(output_dir, 'test_latents.pkl'))
    train_latents = load(join(output_dir, 'train_latents.pkl'))
    for study in test_latents:
        test_latent = test_latents[study]
        train_latent = train_latents[study]
        test_latent -= test_latent.mean(axis=0)
        train_latent -= train_latent.mean(axis=0)
        test_latent_std = test_latent.std(axis=0)
        train_latent_std = train_latent.std(axis=0)

        U, s_train, Vh = svd(train_latent)
        U, s_test, Vh = svd(test_latent)

        fig, ax = plt.subplots(1, 1)
        ax.plot(range(len(s_train)), s_test)
        ax.plot(range(len(s_train)), s_train)
        plt.savefig(join(output_dir, '%s_svd.png' % study))
        plt.close(fig)

        dim = test_latent_std.shape[0]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.bar(range(dim), test_latent_std, width=1)
        ax2.bar(range(dim), train_latent_std, width=1)
        ax1.set_title('Test data')
        ax2.set_title('train data')
        fig.suptitle(study)
        for ax in (ax1, ax2):
            ax.set_xlim([0, dim])
            ax.set_ylim([0, 5])
            ax.set_xlabel('Channel')
            ax.set_ylabel('Mean level')
        plt.savefig(join(output_dir, '%s_latent.png' % study))
        plt.close(fig)


if __name__ == '__main__':
    # compute_components(join(get_output_dir(), 'multi_studies', '57'))
    # compute_components(join(get_output_dir(), 'multi_studies', '107'),
    #                    lstsq=True)
    plot_activation(join(get_output_dir(), 'multi_studies', '901'))
