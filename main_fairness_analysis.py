import itertools
import numpy as np
import argparse
import time

from fairness_analyzer import RfwFairnessAnalyzer, BfwFairnessAnalyzer
from utils import FileManager


def fairness_analysis(datasets, features, approaches, calibration_methods, n_clusters=None, fpr_thr_list=None):
    """
    This function accepts a list of datasets, features, approaches, n_clusters, fpr_ths in order to complete the
    fairness assessment of multiple settings in a single pass.
    """
    # Start time recording
    total_start = time.time()
    # Set the default number of clusters and fpr_thr_list
    if n_clusters is None:
        n_clusters = [100]
    if fpr_thr_list is None:
        fpr_thr_list = [1e-3]
    # Create an experiment manager dictionary
    experiment_runner_dic = {'rfw': RfwFairnessAnalyzer, 'bfw': BfwFairnessAnalyzer}
    # Loop through the datasets
    for dataset in datasets:
        # Instantiate the experiment runner for the dataset
        features_run, calibration_methods, approaches = assign_parameters(
            features, calibration_methods, approaches, experiment_runner_dic[dataset].all_features
        )
        for to_unpack in itertools.product(n_clusters, fpr_thr_list, features_run, approaches, calibration_methods):
            start = time.time()
            n_cluster, fpr_thr, feature, approach, calibration_method = to_unpack
            experiment_runner = experiment_runner_dic[dataset](
                n_cluster, fpr_thr, feature, approach, calibration_method
            )
            print_loop_log(dataset, fpr_thr, feature, approach, calibration_method, n_cluster)
            saveto = FileManager.get_save_file_path(
                dataset, feature, approach, calibration_method, experiment_runner.nbins, n_cluster, fpr_thr
            )
            FileManager.prepare_output_dir(saveto)
            experiment_output = experiment_runner.run_experiment()
            np.save(saveto, experiment_output)
            to_report = (dataset, feature, approach, calibration_method, n_cluster)
            print(f'Analysis for {to_report} took {round(time.time() - start)} seconds!')
    print(f'All experiments took {round(time.time() - total_start)} seconds!')


def assign_parameters(features, calibration_methods, approaches, all_features):
    """
    This function simply updates the parameters of features, calibration_methods and approaches variable such that the
    user can pass 'all' instead of having to explicitly name out all the parameters.
    """
    if features == 'all':
        features = all_features
    elif isinstance(features, str):
        features = [features]
    if approaches == 'all':
        approaches = ['baseline', 'faircal', 'fsn', 'agenda', 'oracle']
    if calibration_methods == 'all':
        calibration_methods = ['binning', 'isotonic_regression', 'beta']
    return features, calibration_methods, approaches


def print_loop_log(dataset, fpr_thr, feature, approach, calibration_method, n_cluster):
    print(f'dataset: {dataset}')
    print('fpr_thr: %.0e' % fpr_thr)
    print('Feature: %s' % feature)
    print('   Approach: %s' % approach)
    print('      Calibration Method: %s' % calibration_method)
    if approach in ('faircal', 'fsn'):
        print('         number clusters: %d' % n_cluster)


def argument_parsing_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets', type=str, nargs='+',
        help='name of dataset, to run both, pass "rfw bfw"',
        choices=['rfw', 'bfw'],
        default=['bfw'])

    parser.add_argument(
        '--features', type=str, nargs='+',
        help='features',
        choices=['facenet', 'facenet-webface', 'arcface', 'all'],
        default='all')

    parser.add_argument(
        '--approaches', type=str, nargs='+',
        help='approaches to use separate using " "',
        choices=['baseline', 'faircal', 'fsn', 'agenda', 'ftc', 'faircal-gmm', 'oracle', 'all'],
        default='all')

    parser.add_argument(
        '--calibration_methods', type=str, nargs='+',
        help='calibration_methods to use, separate using " "',
        choices=['binning', 'isotonic_regression', 'beta', 'all'],
        default=['beta'])

    args = parser.parse_args()

    return args

def run_complete_analysis():
    """
    This function runs the complete fairness analysis under all settings.
    """
    fairness_analysis(
        # datasets=['rfw', 'bfw'],
        datasets=['bfw'],
        # features='all',
        features='facenet-webface',
        # approaches=['baseline', 'faircal', 'fsn', 'agenda', 'faircal-gmm', 'oracle'],
        approaches=['baseline'],
        # approaches=['faircal-gmm'],
        calibration_methods=['beta'],
    )

if __name__ == '__main__':
    # Argument parsing
    args = argument_parsing_func()
    # Run main
    fairness_analysis(
        args.datasets, args.features, args.approaches, args.calibration_methods
    )


