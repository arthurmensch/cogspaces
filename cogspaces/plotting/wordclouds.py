import math
import os
from collections import defaultdict
from itertools import repeat
from os.path import join

from joblib import Parallel, delayed
from wordcloud import WordCloud


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


def rgb2hex(r, g, b):
    return f'#{int(round(r * 255)):02x}' \
           f'{int(round(g * 255)):02x}' \
           f'{int(round(b * 255)):02x}'


def plot_word_clouds(output_dir, grades, n_jobs=1, colors=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if colors is None:
        colors = repeat(None
                        )

    Parallel(n_jobs=n_jobs, verbose=10)(delayed(plot_word_cloud_single)
                                        (output_dir, grades, i, color)
                                        for i, (grades, color) in
                                        enumerate(zip(grades['full'], colors)))


def plot_word_cloud_single(output_dir, grades, index,
                           color=None):
    import seaborn as sns

    if color is not None:
        colormap = sns.dark_palette(color, as_cmap=True)
    else:
        colormap = None

    contrasts = list(filter(
        lambda x: 'effects_of_interest' not in x and 'gauthier' not in x,
        grades))[:15]
    frequencies_cat = defaultdict(lambda: 0.)
    frequencies_single = defaultdict(lambda: 0.)
    occurences = defaultdict(lambda: 0.)
    for contrast in contrasts:
        grade = grades[contrast]
        study, contrast = contrast.split('::')
        contrast = contrast.replace('_', ' ').replace('&', ' ').replace('-',
                                                                        ' ')
        # contrast = filter_contrast(contrast)
        terms = contrast.split(' ')
        cat_terms = []
        for term in terms:
            if term == 'baseline':
                break
            if term in ['vs']:
                break
            cat_terms.append(term)
        for term in cat_terms:
            frequencies_single[term] += grade
            occurences[term] += 1
        cat_terms = ' '.join(cat_terms)
        frequencies_cat[cat_terms] += grade

    frequencies_single = {term: freq / math.sqrt(occurences[term]) for term, freq
                          in frequencies_single.items()}
    width, height = (900, 450)
    wc = WordCloud(prefer_horizontal=1,
                   background_color="rgba(255, 255, 255, 0)",
                   width=width, height=height,
                   colormap=colormap,
                   relative_scaling=0.7)
    wc.generate_from_frequencies(frequencies=frequencies_single, )
    wc.to_file(join(output_dir, 'wc_single_%i.png' % index))

    width, height = (900, 300)

    wc = WordCloud(prefer_horizontal=1,
                   background_color="rgba(255, 255, 255, 0)",
                   width=width, height=height,
                   mode='RGBA',
                   colormap=colormap,
                   relative_scaling=0.8)
    wc.generate_from_frequencies(frequencies=frequencies_cat, )
    wc.to_file(join(output_dir, 'wc_cat_%i.png' % index))

    width, height = (1200, 300)

    wc = WordCloud(prefer_horizontal=1,
                   background_color="rgba(255, 255, 255, 0)",
                   width=width, height=height,
                   mode='RGBA',
                   colormap=colormap,
                   relative_scaling=0.8)
    wc.generate_from_frequencies(frequencies=frequencies_cat, )
    wc.to_file(join(output_dir, 'wc_cat_%i_wider.png' % index))
