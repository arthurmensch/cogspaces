from blaze import join
from nilearn.image import index_img
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map

from cogspaces.datasets.utils import fetch_mask


def maps_from_model(estimator,
                    dictionary,
                    target_encoder,
                    standard_scaler):
    mask = fetch_mask()
    masker = NiftiMasker(mask_img=mask).fit()
    # components.shape = (n_components, n_voxels)
    components = masker.transform(dictionary)
    # coef.shape = (n_components, n_classes)
    coefs = estimator.coefs_
    scale = standard_scaler.scale_
    classes = target_encoder.classes_

    components_dict = {}
    names = {}
    for study in coefs:
        this_scale = scale[study]
        coef = coefs[study]
        these_names = classes[study]
        coef = coef / this_scale[:, None]
        components = coef.T.dot(components)
        components = masker.inverse_transform(components)
        components_dict[study] = components
        names[study] = these_names
    return components, names


def plot_components(imgs, names, output_dir):
    for study in imgs:
        this_img = imgs[study]
        these_names = names[study]

        for i, name in enumerate(these_names):
            full_name = study + ' ' + name
            display = plot_stat_map(index_img(this_img, i),
                                    title=full_name)
            display.save(join(output_dir, '%s.png'
                              % full_name))