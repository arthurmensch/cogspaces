import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from nilearn.plotting import plot_prob_atlas

from cogspaces.datasets import fetch_atlas_modl

data = fetch_atlas_modl()

plot_prob_atlas(data.components128)
plot_prob_atlas(data.components16)
plot_prob_atlas(data.components64)
plt.show()
