import os
import re
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cogspaces.datasets.derivative import get_chance_subjects, \
    get_brainpedia_descr
from cogspaces.datasets.utils import get_output_dir
from matplotlib import gridspec, ticker

idx = pd.IndexSlice


def gather_metrics(output_dir, save_dir):
    regex = re.compile(r'[0-9]+$')
    accuracies = []
    contrasts_metrics = []
    for root, dirs, files in os.walk(output_dir):
        for this_dir in filter(regex.match, dirs):
            this_exp_dir = join(root, this_dir)
            this_dir = int(this_dir)
            try:
                import json
                config = json.load(
                    open(join(this_exp_dir, 'config.json'), 'r'))
                metrics = json.load(
                    open(join(this_exp_dir, 'metrics.json'), 'r'))
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                print('Skipping exp %i' % this_dir)
                continue
            seed = config['system']['seed']
            estimator = config['model']['estimator']
            prec, recall, f1, bacc, accuracy = (metrics['prec'],
                                                metrics['recall'],
                                                metrics['f1'],
                                                metrics['bacc'],
                                                metrics['accuracy'])
            for study, study_accuracy in accuracy.items():
                accuracies.append({'study': study, 'accuracy': study_accuracy,
                                   'seed': seed, 'estimator': estimator})
                for contrast in prec[study]:
                    contrasts_metrics.append(
                        dict(study=study, recall=recall[study][contrast],
                             prec=prec[study][contrast],
                             f1=f1[study][contrast],
                             bacc=bacc[study][contrast], seed=seed,
                             estimator=estimator))
    accuracies = pd.DataFrame(accuracies)
    contrasts_metrics = pd.DataFrame(contrasts_metrics)
    contrasts_metrics.to_pickle(join(save_dir, 'contrasts_metrics.pkl'))

    baseline = 'logistic'
    accuracies.set_index(['estimator', 'study', 'seed'], inplace=True)
    accuracies.sort_index(inplace=True)
    median = accuracies['accuracy'].loc[baseline].groupby(
        level='study').median()
    diffs = []
    baseline_accuracy = accuracies['accuracy'].loc[baseline]
    estimators = accuracies.index.get_level_values('estimator').unique()
    baseline_accuracy = pd.concat([baseline_accuracy] * len(estimators),
                                  keys=estimators, names=['estimator'])
    accuracies['baseline'] = baseline_accuracy
    accuracies['diff_with_baseline'] = accuracies['accuracy'] - accuracies[
        'baseline']
    accuracies['diff_with_baseline_median'] = accuracies['accuracy'].groupby(
        level='study').transform(lambda x: x - median)

    accuracies.to_pickle(join(save_dir, 'accuracies.pkl'))
    return accuracies, contrasts_metrics


def plot_mean_accuracies(save_dir):
    accuracies = pd.read_pickle(join(save_dir, 'accuracies.pkl'))

    data = accuracies.groupby(level=['estimator', 'study']).aggregate(
        ['mean', 'std'])
    data = data.loc['ensemble']
    data = data.sort_values(by=('diff_with_baseline', 'mean'))

    chance, subjects = get_chance_subjects()
    data['chance'] = chance
    print(data)

    width, height = 8.5, 7
    brainpedia = get_brainpedia_descr()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.7, 1])
    fig = plt.figure(figsize=(width, height))
    gs.update(left=0.3, right=0.96, bottom=0.2, top=0.95, wspace=1.2 / width)
    ax2 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax2)
    n_study = len(data)

    ind = np.arange(n_study) * 2 + .5
    width = 1.2
    diff_color = sns.color_palette("husl", n_study)[::-1]
    diff_color_err = [(max(0, r - .1), max(0, g - .1), max(0, b - .1))
                      for r, g, b in diff_color]

    baseline_color = '0.75'
    baseline_color_err = '0.65'
    transfer_color = '0.1'
    transfer_color_err = '0.'
    ax1.barh(ind, data[('diff_with_baseline', 'mean')], width,
             color=diff_color)
    for this_x, this_y, this_xerr, this_color in zip(
            data[('diff_with_baseline', 'mean')],
            ind,
            data[('diff_with_baseline', 'std')],
            diff_color_err):
        ax1.errorbar(this_x, this_y, xerr=this_xerr, elinewidth=1.5,
                     capsize=2, linewidth=0, ecolor=this_color,
                     alpha=.5)
    ax1.set_xlabel('Multi-study acc. gain', fontsize=12)
    ax1.spines['left'].set_position('zero')
    plt.setp(ax1.get_yticklabels(), visible=False)

    # ax1.set_xlim([-0.06, 0.19])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    ind = np.arange(n_study) * 2

    width = .8
    rects1 = ax2.barh(ind, data[('baseline', 'mean')], width,
                      color=baseline_color)
    ax2.errorbar(data[('baseline', 'mean')], ind,
                 xerr=data[('baseline', 'std')],
                 elinewidth=1.5,
                 capsize=2, linewidth=0, ecolor=baseline_color_err,
                 alpha=.5)
    rects2 = ax2.barh(ind + width, data[('accuracy', 'mean')], width,
                      color=transfer_color)
    ax2.errorbar(data[('accuracy', 'mean')], ind + width,
                 xerr=data[('accuracy', 'std')],
                 elinewidth=1.5,
                 capsize=2, linewidth=0, ecolor=transfer_color_err,
                 alpha=.5)

    lines = ax2.vlines([data['chance']],
                       ind - width / 2, ind + 3 * width / 2, colors='r',
                       linewidth=1, linestyles='--')

    ax2.set_ylim([-1, 2 * data.shape[0]])
    plt.setp(ax2.yaxis.get_ticklabels(), fontsize=10)
    # ax2.set_xlim([0., 0.95])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.set_xlabel('Decoding accuracy on test set', fontsize=12)

    handles = [rects1, rects2, lines]
    labels = ['Decoding from\nvoxels',
              'Decoding from\nmulti-study\nnetworks',
              'Chance level']
    ax2.legend(handles, labels, loc='upper left', frameon=False, ncol=3,
               bbox_to_anchor=(0, -.1))

    ax2.set_yticks(ind + width / 2)
    ax2.annotate('Task fMRI study', xy=(-.5, -.08), xycoords='axes fraction',
                 fontsize=12)
    labels = [
        '%s' % brainpedia.loc[label]['Description'] for label in
        data.index.values]
    ax2.set_yticklabels(labels, ha='right', va='center', fontsize=8.5)
    sns.despine(fig)
    # plt.show()
    plt.savefig(join(save_dir, 'mean_accuracies.pdf'))
    plt.close(fig)


def plot_accuracies(save_dir):
    width, height = 6.2, 3

    accuracies = pd.read_pickle(join(save_dir, 'accuracies.pkl'))

    mean_accuracies = accuracies.groupby(
        level=['estimator', 'study']).aggregate(['mean', 'std'])
    mean_accuracies = mean_accuracies.loc['multi_study']
    mean_accuracies = mean_accuracies.sort_values(
        by=('diff_with_baseline', 'mean'))
    sort = mean_accuracies.index.get_level_values('study')

    accuracies = accuracies.reindex(index=sort, level='study')
    accuracies = accuracies.reindex(index=['logistic', 'multi_study', 'ensemble'],
                                    level='estimator')
    accuracies = accuracies.reset_index()
    n_studies = len(sort)

    diff_color = sns.color_palette("husl", n_studies)
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    fig.subplots_adjust(left=.05, right=0.7,
                        bottom=0.25, top=1)

    params = dict(x="diff_with_baseline_median", y="estimator", hue="study",
                  data=accuracies, dodge=True, ax=ax,
                  palette=diff_color
                  )
    sns.stripplot(alpha=1, zorder=200, size=3.5, linewidth=0, jitter=True,
                  **params)
    sns.boxplot(x="diff_with_baseline_median", y="estimator", color='0.75',
                showfliers=False,
                whis=1.5,
                data=accuracies, ax=ax, zorder=100)
    ax.set_xlim([-0.14, 0.175])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel('Accuracy gain compared to baseline median')

    ax.set_ylabel('')
    ax.spines['left'].set_position('zero')
    plt.setp(ax.spines['left'], zorder=2)
    ax.set_yticklabels([])
    ax.set_yticks([])

    estimators = accuracies['estimator'].unique()

    y_labels = {}
    y_labels['multi_study'] = ('Decoding from\nmulti-study\ntask-optimized\n'
                               'networks')
    y_labels['ensemble'] = ('Decoding from\nmulti-study\ntask-optimized\n'
                            'networks (ensemble)')
    y_labels['logistic'] = ('Standard decoding\nfrom resting-state\n'
                            'loadings')

    for i, estimator in enumerate(estimators):
        if i == len(estimators) - 1:
            fontweight = 'bold'
        else:
            fontweight = 'normal'
        ax.annotate(y_labels[estimator], xy=(1, i), xytext=(10, 0),
                    textcoords="offset points",
                    fontweight=fontweight,
                    xycoords=('axes fraction', 'data'),
                    va='center',
                    ha='left')

    sns.despine(fig)

    plt.setp(ax.legend(), visible=False)
    plt.savefig(join(save_dir, 'accuracies.pdf'))
    plt.close(fig)


output_dir = get_output_dir(output_dir=None)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_dir = join(output_dir, 'compare')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

gather_metrics(output_dir=output_dir, save_dir=save_dir)
plot_mean_accuracies(save_dir=save_dir)
plot_accuracies(save_dir)
