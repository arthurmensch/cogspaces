import os
import shutil
from jinja2 import Template
from os.path import join

from cogspaces.datasets.utils import get_output_dir

# Only valid for output_dir above
select = {14: 'Left motor',
          5: 'Right motor',
          3: 'Fusiform gyrus',
          30: 'Lateral occipital',
          # 127: 'Dorsal secundary visual',
          87: 'Primary visual',
          37: 'Right posterior insula',
          4: 'Auditory',
          16: 'Language',
          77: 'Left DLPFC (+ IPS)',
          58: 'Anterior insula',
          6: 'ACC',
          44: 'Cerebellum',}

if __name__ == '__main__':
    output_dir = join(get_output_dir(), '/home/arthur/output/cogspaces/'
                                        'factored_refit_gm_normal_init_'
                                        'full_rest_positive_notune/3')

    wc_dir = join(output_dir, 'wc_cosine_similarities')
    components_dir = join(output_dir, 'components')

    copies = []

    selected_dir = join(output_dir, 'selected_components')

    if not os.path.exists(selected_dir):
        os.makedirs(join(output_dir, selected_dir))
    imgs = []

    for index, name in select.items():
        wc_single = join(wc_dir, 'wc_single_%i.png' % index)
        wc_cat = join(wc_dir, 'wc_cat_%i.png' % index)
        stat_map = join(components_dir, 'components_%i_stat_map.png' % index)
        glass_brain = join(components_dir, 'components_%i_glass_brain.png' % index)
        these_imgs = []
        for src in [stat_map, glass_brain, wc_single, wc_cat]:
            dirname, filename = os.path.split(src)
            name_curated = name.lower().replace(' ', '_').replace('/', '_')
            filename = filename.replace('components_%i' % index, name_curated)
            filename = filename.replace('wc_cat_%i' % index, '%s_wc_cat' % name_curated)
            filename = filename.replace('wc_single_%i' % index, '%s_wc_single' % name_curated)
            tgt = join(selected_dir, filename)
            these_imgs.append(filename)
            copies.append((src, tgt))
            if src in [stat_map, glass_brain]:
                copies.append((src.replace('.png', '.svg'), tgt.replace('.png', '.svg')))
        imgs.append((these_imgs, name))

    for src, tgt in copies:
        shutil.copy(src, tgt)

    with open('plot_maps.html', 'r') as f:
        template = f.read()
    template = Template(template)
    html = template.render(imgs=imgs)
    output_file = join(selected_dir, 'components.html')
    with open(output_file, 'w+') as f:
        f.write(html)
