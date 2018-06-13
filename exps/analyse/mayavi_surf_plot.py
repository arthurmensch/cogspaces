# Run in "ipython --matplotlib=qt"
import numpy as np
from mayavi import mlab
from nilearn import datasets, image
from nilearn import surface
from os.path import expanduser

fsaverage = datasets.fetch_surf_fsaverage5()

def plot_on_surf(data, sides=['left', 'right'],
                 threshold=None, inflate=1.001, **kwargs):
    """ Plot a numpy array of data on the corresponding fsaverage
        surface.

        The kwargs are passed to mlab.triangular_mesh
    """
    for side in sides:
        mesh = surface.load_surf_mesh(fsaverage['infl_%s' % side])
        shift = -35 if side == 'left' else 35
        surf = mlab.triangular_mesh(inflate * mesh[0][:, 0] + shift,
                                    inflate * mesh[0][:, 1],
                                    inflate * mesh[0][:, 2],
                                    mesh[1],
                                    scalars=data,
                                    **kwargs,
                                    )
        # Avoid openGL bugs with backfaces
        surf.actor.property.backface_culling = True
        # Smoother rendering
        surf.module_manager.source.filter.splitting = False

        if threshold is not None:
            surf.enable_contours = True
            surf.contour.auto_contours = False
            surf.contour.contours = [threshold, data.max()]
            # Add a second surface to fill the contours
            surf2 = mlab.pipeline.surface(surf, color=kwargs['color'],
                                          opacity=.5)
            surf2.enable_contours = True
            surf2.contour.auto_contours = False
            surf2.contour.contours = [threshold, data.max()]
            surf2.contour.filled_contours = True


def add_surf_map(niimg, sides=['left', 'right'],
                 threshold=None, **kwargs):
    """ Project a volumetric data and plot it on the corresponding
        fsaverage surface.

        The kwargs are passed to mlab.triangular_mesh
    """
    for side in sides:
        data = surface.vol_to_surf(niimg, fsaverage['pial_%s' % side])
        plot_on_surf(data, sides=sides, threshold=threshold, **kwargs)



###############################################################################


fig = mlab.figure(bgcolor=(1, 1, 1))

# Disable rendering to speed things up
fig.scene.disable_render = True

# Plot the background
for side in ['left', ]:#'right']:
    depth = surface.load_surf_data(fsaverage['sulc_%s' % side])
    plot_on_surf(depth, sides=[side, ], colormap='gray', inflate=.995)

rng = np.random.RandomState(42)

components = image.load_img(expanduser('~/output/cogspaces/single_full/components.nii.gz'))
mask = datasets.load_mni152_brain_mask().get_data() > 0

n_components = components.shape[-1]
threshold = np.percentile(np.abs(components.get_data()[mask]),
                          100. * (1 - 1. / n_components))

# To speed up when prototyping
#components = image.index_img(components, slice(0, 10))

for i, component in enumerate(image.iter_img(components)):
    print('Component %i' % i)
    component = image.math_img('np.abs(img)', img=component)
    add_surf_map(component, threshold=threshold,
                 color=tuple(rng.random_sample(size=3)),
                 sides=['left', ])

# Side view
#mlab.view(5.5, 75.5, 527.5)
# Side & front
mlab.view(55, 73, 527.5)

# Enable rendering
fig.scene.disable_render = False
