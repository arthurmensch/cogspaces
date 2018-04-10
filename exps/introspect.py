import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from nilearn.decomposition import CanICA
from nilearn.image import index_img, iter_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from os.path import join, expanduser
from scipy.linalg import svd
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.utils.extmath import randomized_svd

from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import fetch_mask, get_data_dir
from exps.train import load_data


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


def compute_latent(output_dir, lstsq):
    estimator = load(join(output_dir, 'estimator.pkl'))

    modl_atlas = fetch_atlas_modl()
    dictionary = modl_atlas['components512']
    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    dictionary = masker.transform(dictionary)
    if lstsq:
        gram = dictionary.dot(dictionary.T)
        dict_proj = np.linalg.inv(gram).dot(dictionary)
    else:
        dict_proj = dictionary
    sup_proj = estimator.module_.embedder.linear.weight.data.numpy()
    proj = sup_proj @ dict_proj
    gram = proj @ proj.T
    back_proj = np.linalg.inv(gram) @ proj
    print(back_proj)
    print(back_proj.shape)

    # module = estimator.module_
    # latent_weight = module.embedder.linear.weight.data.cpu().numpy()
    # projector = latent_weight.dot(dictionary)
    #
    # components, variance, _ = randomized_svd(projector.T, n_components=30)
    # _, _, sources = fastica(components, whiten=True, fun='cube')
    # img = masker.inverse_transform(sources.T)
    # # for i, this_img in enumerate(iter_img(img)):
    # #     fig, ax = plt.subplots(1, 1)
    # #     plot_stat_map(this_img, fig=fig, ax=ax)
    # #     plt.close(fig)
    # #     plt.savefig(expanduser('~/components_%i.nii.gz' % i))
    # img.to_filename(expanduser('~/components_orth.nii.gz'))

    source_dir = join(get_data_dir(), 'reduced_512_lstsq')
    data, target = load_data(source_dir, 'all', 'archi')
    data = {'archi': data['archi']}
    latents = estimator.predict_latent(data)['archi']
    rec = latents.dot(back_proj)
    print(data)
    print(rec)




    # print(projector.shape)
    # fast_ica = FastICA(whiten=False, algorithm='deflation')
    # fast_ica.fit(projector)
    # projector = fast_ica.components_
    # img = masker.inverse_transform(projector)
    # img.to_filename(expanduser('~/components.nii.gz'))

#
#
# def compute_components(output_dir, lstsq):
#     estimator = load(join(output_dir, 'estimator.pkl'))
#     target_encoder = load(join(output_dir, 'target_encoder.pkl'))
#     standard_scaler = load(join(output_dir, 'standard_scaler.pkl'))
#
#     modl_atlas = fetch_atlas_modl()
#     dictionary = modl_atlas['components512']
#     plot_dir = join(output_dir, 'plot')
#     if not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)
#     dump(names, join(plot_dir, 'names.pkl'))
#     for study, this_components in components.items():
#         this_components.to_filename(join(plot_dir,
#                                          'components_%s.nii.gz' % study))
#         plot_components(components, names, plot_dir)
#
#

def plot_activation(output_dir):
    test_latents = load(join(output_dir, 'test_latents.pkl'))
    train_latents = load(join(output_dir, 'train_latents.pkl'))
    test_latent_all = np.concatenate(list(test_latents.values()))
    train_latent_all = np.concatenate(list(train_latents.values()))

    U, s_train, Vh = svd(test_latent_all)
    U, s_test, Vh = svd(train_latent_all)

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(s_test)), s_test)
    ax.plot(range(len(s_train)), s_train)
    plt.savefig(join(output_dir, 'all_svd.png'))
    plt.close(fig)

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
        ax.plot(range(len(s_test)), s_test)
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
    compute_latent(expanduser('~/322'), True)
    # compute_components(join(get_output_dir(), 'multi_studies', '107'),
    #                    lstsq=True)
    # plot_activation(join(get_output_dir(), 'multi_studies', '922'))
