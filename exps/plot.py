from os.path import join

import numpy as np
from joblib import Memory
from matplotlib.testing.compare import get_cache_dir
from seaborn import hls_palette
from sklearn.utils import check_random_state

from cogspaces.plotting.volume import plot_4d_image
from cogspaces.report import get_names, components_html, \
    classifs_html, compute_nifti

mem = Memory(cachedir=get_cache_dir())

n_jobs = 3
output_dir = 'output'
rng = check_random_state(1000)

# Compute components
classifs_imgs, components_imgs = compute_nifti(output_dir)
names, full_names = get_names(output_dir)

# Colors
colors = np.arange(128)
colors_2d = np.array(hls_palette(128, s=1, l=.4))
colors_word_cl = np.array(hls_palette(128, s=.7, l=.4))
colors_3d = np.array(hls_palette(128, s=1, l=.5))

np.save(join(output_dir, 'colors_2d.npy'), colors_2d)
np.save(join(output_dir, 'colors_3d.npy'), colors_3d)

# 2D plots
view_types = ['stat_map', 'glass_brain', ]
plot_4d_image(classifs_imgs,
              output_dir=join(output_dir, 'classifs'),
              names=full_names,
              view_types=view_types, threshold=0,
              n_jobs=n_jobs)
plot_4d_image(components_imgs,
              output_dir=join(output_dir, 'components'),
              names='components',
              colors=colors_2d,
              view_types=view_types,
              n_jobs=n_jobs)

# 3D plots
# from cogspaces.plotting.surface import plot_4d_image_surface
# plot_4d_image_surface(components_imgs, colors_3d, output_dir)

# Wordclouds
# grades = compute_grades(output_dir, grade_type='cosine_similarities')
# dump(grades, join(output_dir, 'grades.pkl'))
# from cogspaces.plotting.wordclouds import plot_word_clouds
# plot_word_clouds(join(output_dir, 'wc'), grades, n_jobs=n_jobs, colors=colors)

# HTML report
components_html(output_dir, 'components')
classifs_html(output_dir, 'classifs')
