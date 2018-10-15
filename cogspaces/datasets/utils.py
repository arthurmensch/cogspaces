import os
import re


def get_data_dir(data_dir=None):
    """ Returns the directories in which to look for data.

    Parameters
    ----------
    data_dir: string, optional
        Path of the utils directory. Used to force utils storage in a specified
        location. Default: None

    Returns
    -------
    path: string
        Path of the data directory.
    """

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        assert (isinstance(data_dir, str))
        return data_dir
    elif 'COGSPACES_DATA' in os.environ:
        return os.environ['COGSPACES_DATA']
    else:
        return os.path.expanduser('~/cogspaces_data')


def get_output_dir(output_dir=None):
    """ Returns the directories in which to save output.

    Parameters
    ----------
    output_dir: string, optional
        Path of the utils directory. Used to force utils storage in a specified
        location. Default: None

    Returns
    -------
    path: string
        Path of the output directory.
    """

    # Check data_dir which force storage in a specific location
    if output_dir is not None:
        assert (isinstance(output_dir, str))
        return output_dir
    elif 'COGSPACES_OUTPUT' in os.environ:
        return os.environ['COGSPACES_OUTPUT']
    else:
        return os.path.expanduser('~/cogspaces_data')


def filter_contrast(contrast):
    contrast = contrast.lower()
    contrast = contrast.replace('lf', 'left foot')
    contrast = contrast.replace('rf', 'right foot')
    contrast = contrast.replace('lh', 'left hand')
    contrast = contrast.replace('rh', 'right hand')
    contrast = contrast.replace('clicgaudio', 'left audio click')
    contrast = contrast.replace('clicgvideo', 'left video click')
    contrast = contrast.replace('clicdvideo', 'left video click')
    contrast = contrast.replace('clicdaudio', 'right audio click')
    contrast = contrast.replace('calculvideo', 'video calculation')
    contrast = contrast.replace('calculaudio', 'audio calculation')
    contrast = contrast.replace('damier h', 'horizontal checkerboard')
    contrast = contrast.replace('damier v', 'vertical checkerboard')

    contrast = contrast.replace('audvid600', 'audio video 600ms')
    contrast = contrast.replace('audvid1200', 'audio video 1200ms')
    contrast = contrast.replace('audvid300', 'audio video 300ms')
    contrast = contrast.replace('bk', 'back')
    contrast = contrast.replace('realrt', 'real risk-taking')
    contrast = contrast.replace('reapp', 'reappraise')
    contrast = re.sub(r'\b(rt)\b', 'risk-taking', contrast)
    contrast = re.sub(r'\b(ons)\b', '', contrast)
    contrast = re.sub(r'\b(neu)\b', 'neutral', contrast)
    contrast = re.sub(r'\b(neg)\b', 'negative', contrast)
    contrast = re.sub(r'\b(ant)\b', 'anticipated', contrast)
    return contrast