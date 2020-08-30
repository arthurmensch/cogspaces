"""Perform traininig of a multi-study model using the fetchers provided by cogspaces.

Hyperparameters can be edited in the file."""

import argparse
import json
import os
from os.path import join

import numpy as np
from joblib import Memory, dump
from sklearn.metrics import accuracy_score

from cogspaces.classification.ensemble import EnsembleClassifier
from cogspaces.classification.logistic import MultiLogisticClassifier
from cogspaces.classification.multi_study import MultiStudyClassifier
from cogspaces.datasets import STUDY_LIST, load_reduced_loadings
from cogspaces.datasets.contrast import load_masked_contrasts
from cogspaces.datasets.derivative import load_from_directory, split_studies, get_study_info
from cogspaces.datasets.utils import get_output_dir
from cogspaces.model_selection import train_test_split
from cogspaces.preprocessing import MultiStandardScaler, MultiTargetEncoder
from cogspaces.utils import compute_metrics, ScoreCallback, MultiCallback


def run(estimator='multi_study', seed=0, plot=False, n_jobs=1, use_gpu=False, split_by_task=False,
        verbose=0):
    # Parameters
    system = dict(
        verbose=verbose,
        n_jobs=n_jobs,
        plot=plot,
        seed=seed,
        output_dir=None
    )
    data = dict(
        studies='all',
        dataset='loadings',  # Useful to override source directory
        test_size=0.5,
        train_size=20,
        reduced=True,
        data_dir=None,
    )
    model = dict(
        split_by_task=split_by_task,
        estimator=estimator,
        normalize=True,
        seed=100,
        target_study=None,
    )

    config = {'system': system, 'data': data, 'model': model}

    if model['estimator'] in ['multi_study', 'ensemble']:
        multi_study = dict(
            latent_size=128,
            weight_power=0.6,
            batch_size=128,
            init='resting-state',
            latent_dropout=0.75,
            input_dropout=0.25,
            device='cuda:0' if use_gpu else 'cpu',
            seed=100,
            lr={'pretrain': 1e-3, 'train': 1e-3, 'finetune': 1e-3},
            max_iter={'pretrain': 200, 'train': 300, 'finetune': 200},
        )
        config['multi_study'] = multi_study
        if model['estimator'] == 'ensemble':
            ensemble = dict(
                seed=100,
                n_runs=120,
                alpha=1e-5, )
            config['ensemble'] = ensemble
    else:
        logistic = dict(l2_penalty=np.logspace(-7, 0, 4).tolist(),
                        max_iter=1000, )
        config['logistic'] = logistic

    output_dir = join(get_output_dir(config['system']['output_dir']),
                      'low_subjects',
                      'split_by_task' if split_by_task else 'split_by_study',
                      config['model']['estimator'],
                      str(config['system']['seed']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    info = {}

    with open(join(output_dir, 'config.json'), 'w+') as f:
        json.dump(config, f)

    print("Loading data")
    if data['studies'] == 'all':
        studies = STUDY_LIST
    elif isinstance(data['studies'], str):
        studies = [data['studies']]
    elif isinstance(data['studies'], list):
        studies = data['studies']
    else:
        raise ValueError("Studies should be a list or 'all'")

    if data['dataset'] is not None:
        input_data, targets = load_from_directory(dataset=data['dataset'], data_dir=data['data_dir'])
    elif data['reduced']:
        input_data, targets = load_reduced_loadings(data_dir=data['data_dir'])
    else:
        input_data, targets = load_masked_contrasts(data_dir=data['data_dir'])

    if isinstance(data['train_size'], int):
        n_subjects = get_study_info().groupby(by='study').first()['#subjects']
        studies = list(filter(lambda study: data['train_size'] < n_subjects.loc[study] // 2, studies))
    print(studies)

    input_data = {study: input_data[study] for study in studies}
    targets = {study: targets[study] for study in studies}

    if model['split_by_task']:
        _, split_targets = split_studies(input_data, targets)
    else:
        split_targets = targets
    target_encoder = MultiTargetEncoder().fit(split_targets)

    train_data, test_data, train_targets, test_targets = \
        train_test_split(input_data, targets, random_state=system['seed'],
                         test_size=data['test_size'],
                         train_size=data['train_size'])
    if model['split_by_task']:
        train_data, train_targets = split_studies(train_data, train_targets)
        test_data, test_targets = split_studies(test_data, test_targets)
    train_targets = target_encoder.transform(train_targets)
    test_targets = target_encoder.transform(test_targets)

    print("Setting up model")
    if model['normalize']:
        standard_scaler = MultiStandardScaler().fit(train_data)
        train_data = standard_scaler.transform(train_data)
        test_data = standard_scaler.transform(test_data)
    else:
        standard_scaler = None

    if model['estimator'] in ['multi_study', 'ensemble']:
        estimator = MultiStudyClassifier(verbose=system['verbose'],
                                         n_jobs=system['n_jobs'],
                                         **multi_study)
        if model['estimator'] == 'ensemble':
            memory = Memory(cachedir=None)
            estimator = EnsembleClassifier(estimator,
                                           n_jobs=system['n_jobs'],
                                           memory=memory,
                                           **ensemble
                                           )
            callback = None
        else:
            # Set some callback to obtain useful verbosity
            test_callback = ScoreCallback(Xs=test_data, ys=test_targets,
                                          score_function=accuracy_score)
            train_callback = ScoreCallback(Xs=train_data, ys=train_targets,
                                           score_function=accuracy_score)
            callback = MultiCallback({'train': train_callback,
                                      'test': test_callback})
            info['n_iter'] = train_callback.n_iter_
            info['train_scores'] = train_callback.scores_
            info['test_scores'] = test_callback.scores_
    elif model['estimator'] == 'logistic':
        estimator = MultiLogisticClassifier(verbose=system['verbose'],
                                            n_jobs=n_jobs,
                                            **logistic)
        callback = None

    print("Training model")
    estimator.fit(train_data, train_targets, callback=callback)

    print("Evaluating model")
    test_preds = estimator.predict(test_data)
    metrics = compute_metrics(test_preds, test_targets, target_encoder)
    print(metrics['accuracy'])

    print("Saving model")
    # Save model for further analysis
    dump(target_encoder, join(output_dir, 'target_encoder.pkl'))
    if model['normalize']:
        dump(standard_scaler, join(output_dir, 'standard_scaler.pkl'))
    dump(estimator, join(output_dir, 'estimator.pkl'))
    with open(join(output_dir, 'metrics.json'), 'w+') as f:
        json.dump(metrics, f)
    with open(join(output_dir, 'info.json'), 'w+') as f:
        json.dump(info, f)

    if config['system']['plot']:
        from utils.plotting import make_plots, prepare_plots
        print('Preparing plots')
        prepare_plots(output_dir)
        print("Plotting model")
        plot_components = config['model']['estimator'] in ['multi_study',
                                                           'ensemble']
        make_plots(output_dir, plot_classifs=True,
                   plot_components=plot_components,
                   plot_surface=False, plot_wordclouds=True,
                   n_jobs=config['system']['n_jobs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--estimator', type=str,
                        choices=['logistic', 'multi_study', 'ensemble'],
                        default='multi_study',
                        help='estimator type')
    parser.add_argument('-s', '--seed', type=int,
                        default=0,
                        help='Integer to use to seed the model and half-split cross-validation')
    parser.add_argument('-v', '--verbose', type=int,
                        default=0,
                        help='Verbosity')
    parser.add_argument('-p', '--plot', action="store_true",
                        help='Plot the results (classification maps, cognitive components)')
    parser.add_argument('-t', '--task', action="store_true",
                        help='Split by tasks')
    parser.add_argument('-g', '--gpu', action="store_true",
                        help='Split by tasks')
    parser.add_argument('-j', '--n_jobs', type=int,
                        default=1, help='Number of CPUs to use')
    args = parser.parse_args()

    run(args.estimator, args.seed, args.plot, args.n_jobs, args.gpu, args.task, args.verbose)
