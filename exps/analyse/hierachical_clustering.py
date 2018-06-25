import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import load, Memory, delayed, Parallel, dump
from matplotlib.testing.compare import get_cache_dir
from nilearn.input_data import NiftiMasker
from os.path import join
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram

from cogspaces.datasets.utils import get_output_dir, fetch_mask

method = 'average'
os.chdir(join(get_output_dir(), 'figure_4', 'classifs'))
compute = True
plot = True


def cosine_dist(comp):
    S = np.sqrt(np.sum(comp ** 2, axis=1))
    x = 1 - (comp.dot(comp.T)
             / S[:, None]
             / S[None, :])
    return x, x[np.triu_indices(x.shape[0], k=1)]


def analyse_classif(model, classif, keyword=None):
    names = load('names.pkl')
    names = np.array(names)
    if keyword is not None:
        kw_in = np.vectorize(lambda x: any(k in x for k in keyword))
        mask = kw_in(names)
        indices = np.nonzero(mask)[0].tolist()
        classif = classif[indices]
        names = names[indices]
        keyword = '_'.join(keyword)
    else:
        keyword = 'none'
    corr, y = cosine_dist(classif)
    Z = linkage(y, method=method, optimal_ordering=True)
    c, coph_dists = cophenet(Z, y)
    d = dendrogram(Z, no_plot=True, get_leaves=True)
    sort = d['leaves']
    names = np.array(names)[sort]
    corr = corr[sort][:, sort]
    mean_corr = np.mean(np.abs(corr))
    return model, keyword, names, corr, Z, c, mean_corr


if compute:
    mem = Memory(cachedir=get_cache_dir())

    mask = fetch_mask()['hcp']
    masker = NiftiMasker(mask_img=mask, memory=mem).fit()

    classifs = {'full': masker.transform('classifs_full.nii.gz'),
                'factored': masker.transform('classifs_factored.nii.gz'),
                'logistic': masker.transform('classifs_logistic.nii.gz')}

    keywords = [None, ('archi', 'brainomics'), ('house', 'tool')]

    res = Parallel(n_jobs=3)(delayed(mem.cache(analyse_classif))(model, classif, keyword)
                             for model, classif in classifs.items()
                             for keyword in keywords)
    for model, keyword, names, corr, Z, c in res:
        dump((names, corr, Z, c), 'dendrogram_data_%s_%s.pkl' % (model, keyword))

if plot:
    for model in ['full', 'logistic', 'factored']:
        for keyword in None, ('archi', 'brainomics'), ('house', 'tool', 'face'):
            if keyword is None:
                keyword = 'none'
            else:
                keyword = '_'.join(keyword)

            names, corr, Z, c, mean_corr = load('dendrogram_data_%s_%s.pkl' % (model, keyword))
            # Compute and plot first dendrogram.
            fig = plt.figure(figsize=(12, 8))
            fig.subplots_adjust(right=1 - 4 / 12)
            ax1 = fig.add_axes([0.06, 0.1, 0.4 / 3, 0.6])
            Z1 = dendrogram(Z, orientation='left')
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Compute and plot second dendrogram.
            ax2 = fig.add_axes([0.2, 0.71, 0.4, 0.2])
            Z2 = dendrogram(Z)
            ax2.set_xticks([])
            ax2.set_yticks([])

            # Plot distance matrix.
            axmatrix = fig.add_axes([0.2, 0.1, 0.4, 0.6])
            im = axmatrix.matshow(corr, aspect='auto', origin='lower', vmax=2, vmin=0)
            axmatrix.set_xticks([])
            axmatrix.yaxis.tick_right()
            axmatrix.yaxis.set_label_position("right")
            if keyword is not 'none':
                axmatrix.set_yticks(range(len(corr)))
                axmatrix.set_yticklabels(names)
            else:
                axmatrix.set_yticks([])

            axmatrix.annotate('Mean corr: %.3f' % mean_corr, xy=(.5, -.07),
                              xycoords='axes fraction')
            axmatrix.annotate('Cophenetic coeficient: %.3f' % c, xy=(.5, -.1),
                              xycoords='axes fraction')
            axmatrix.annotate('Model : %s' % model, xy=(.5, -.13),

                              xycoords='axes fraction')
            # Plot colorbar.
            # axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
            # plt.colorbar(im, cax=axcolor)
            fig.savefig('dendrogram_%s_%s.png' % (model, keyword))
