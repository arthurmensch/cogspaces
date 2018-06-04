import matplotlib.pyplot as plt
from cogspaces.datasets.utils import fetch_mask, get_output_dir
from joblib import load, Memory
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map, find_xyz_cut_coords
from os.path import join, expanduser

output_dir = join(get_output_dir(), 'compare')

interests = {'right': ['right', 'RH', 'RF'],
             'language': ['_sentence'],
             'left': ['left', 'LH', 'LF'],
             'computation': ['calculation', 'computation', 'calcul'],
             'audio': ['audio', 'auditory']}


def fetch_maps(interest):
    classif = join(output_dir, 'classif.nii.gz')
    classif_ = join(output_dir, 'classif_baseline.nii.gz')
    names = load(join(output_dir, 'names.pkl'))

    idx_names = list(filter(lambda x: check_belongs_to(interest, x[1]),
                            enumerate(names)))

    print(idx_names)
    indices, filtered_names = zip(*idx_names)
    indices = list(indices)

    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask).fit()
    X = masker.transform(classif)
    X = X[indices]
    X_ = masker.transform(classif_)
    X_ = X_[indices]
    return masker, X, X_, filtered_names


def check_belongs_to(l, s):
    s = s.lower()
    return sum([e in s for e in l]) > 0


def main():
    introspect_dir = join(get_output_dir(), 'compare')
    mem = Memory(cachedir=expanduser('~/cache'))

    for name, interest in interests.items():
        masker, X, X_, filtered_names = mem.cache(fetch_maps)(interest)
        mean = X.mean(axis=0)
        mean_ = X_.mean(axis=0)

        img = masker.inverse_transform(mean[None, :])
        img_ = masker.inverse_transform(mean_[None, :])

        vmax = img.get_data().max()
        cut_coords = find_xyz_cut_coords(img, activation_threshold=vmax / 3)

        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        plot_stat_map(img, cut_coords=cut_coords, figure=fig,
                      axes=axes[0], threshold=0, title='factored')
        plot_stat_map(img_, cut_coords=cut_coords,
                      figure=fig, axes=axes[1],
                      threshold=0, title='baseline')
        fig.savefig(join(introspect_dir, '%s.png' % name))
        plt.close(fig)


if __name__ == '__main__':
    main()
