import pandas as pd
import numpy as np
import os
import argparse
import torch
import pickle
import time

from approaches import baseline
from approaches import cluster_methods
from approaches_ftc import ftc
from approaches_agenda import agenda
from approaches import oracle
from utils import prepare_dir
from utils import compute_scores
from sklearn.metrics import roc_curve



experiments_folder = 'experiments/'
ftc_settings_folder = 'experiments/ftc_settings/'
agenda_settings_folder = 'experiments/agenda_settings/'


def gather_results(dataset_name, db_input, nbins, n_clusters, fpr_thr, feature, approach, calibration_method, embedding_data):

    db = db_input.copy()
    if dataset_name == 'rfw':
        # RFW subgroup
        subgroups = {'ethnicity': ['African', 'Asian', 'Caucasian', 'Indian']}
        # RFW sensitive attributes
        sensitive_attributes = {'ethnicity': ['ethnicity', 'ethnicity']}
        # Assigning the embedding to each image
        db['image_id_1_clean'] = db['id1'].map(str) + '_000' + db['num1'].map(str)
        db['image_id_2_clean'] = db['id2'].map(str) + '_000' + db['num2'].map(str)
        embedding_map = dict(zip(embedding_data['image_id'], embedding_data['embedding']))
        db['emb_1'] = db['image_id_1_clean'].map(embedding_map)
        db['emb_2'] = db['image_id_2_clean'].map(embedding_map)
        # Create a mask to remove the rows with missing cosine similarity or embedding.
        keep_cond = (
                db[feature].notnull() &
                db['emb_1'].notnull() &
                db['emb_2'].notnull()
        )
    elif dataset_name == 'bfw':
        # BFW subgroup
        subgroups = {
            'e': ['B', 'A', 'W', 'I'],
            'g': ['F', 'M'],
            'att': ['asian_females', 'asian_males', 'black_males', 'black_females', 'white_females', 'white_males',
                    'indian_females', 'indian_males']
        }
        # BFW sensitive attributes
        sensitive_attributes = {'e': ['e1', 'e2'], 'g': ['g1', 'g2'], 'att': ['att1', 'att2']}
        # Assigning the embedding to each image
        embedding_map = dict(zip(embedding_data['img_path'], embedding_data['embedding']))
        db['emb_1'] = db['path1'].map(embedding_map)
        db['emb_2'] = db['path2'].map(embedding_map)

        # Create a mask to remove the rows with missing cosine similarity or embedding.
        keep_cond = (
                db[feature].notnull() &
                db['emb_1'].notnull() &
                db['emb_2'].notnull()
        )
    else:
        raise ValueError(f'Unrecognised dataset_name: {dataset_name}')

    # remove image pairs that have missing cosine similarities
    db = db[keep_cond].reset_index(drop=True)

    data = {}

    # select one of the folds to be the test set
    for i_variable, fold in enumerate([1, 2, 3, 4, 5]):
        db_fold = {'cal': db[db['fold'] != fold], 'test': db[db['fold'] == fold]}
        scores = {}
        ground_truth = {}
        subgroup_scores = {}
        for dataset in ['cal', 'test']:
            scores[dataset] = np.array(db_fold[dataset][feature])
            ground_truth[dataset] = np.array(db_fold[dataset]['same'])
            subgroup_scores[dataset] = {}
            for att in subgroups.keys():
                subgroup_scores[dataset][att] = {}
                subgroup_scores[dataset][att]['left'] = np.array(db_fold[dataset][sensitive_attributes[att][0]])
                subgroup_scores[dataset][att]['right'] = np.array(db_fold[dataset][sensitive_attributes[att][1]])

        if approach == 'baseline':
            confidences = baseline(scores, ground_truth, nbins, calibration_method)
        elif approach == 'faircal':
            scores, ground_truth, confidences, fair_scores = cluster_methods(
                nbins,
                calibration_method,
                dataset_name,
                feature,
                fold,
                db_fold,
                n_clusters,
                False,
                0,
                embedding_data
            )
        elif approach == 'fsn':
            scores, ground_truth, confidences, fair_scores = cluster_methods(
                nbins,
                calibration_method,
                dataset_name,
                feature,
                fold,
                db_fold,
                n_clusters,
                True,
                fpr_thr,
                embedding_data
            )
        elif approach == 'ftc':
            fair_scores, confidences, model = ftc(dataset_name, feature, db_fold, nbins, calibration_method)
            saveto = '_'.join(
                [dataset_name, calibration_method, feature, 'fold', str(fold)])
            saveto = ftc_settings_folder + saveto
            prepare_dir(saveto)
            torch.save(model.state_dict(), saveto)
        elif approach == 'agenda':
            fair_scores, confidences, modelM, modelC, modelE = agenda(
                dataset_name, feature, db_fold, nbins, calibration_method, embedding_data
            )
            saveto = '_'.join(
                [dataset_name, calibration_method, feature, 'fold', str(fold)])
            saveto = agenda_settings_folder + saveto
            prepare_dir(saveto)
            torch.save(modelM.state_dict(), saveto+'_modelM')
            torch.save(modelC.state_dict(), saveto+'_modelC')
            torch.save(modelE.state_dict(), saveto+'_modelE')
        elif approach == 'oracle':
            confidences = oracle(scores, ground_truth, subgroup_scores, subgroups, nbins, calibration_method)
        else:
            raise ValueError('Approach %s not available.' % approach)

        fpr = {}
        tpr = {}
        thresholds = {}
        ece = {}
        ks = {}
        brier = {}

        for att in subgroups.keys():
            fpr[att] = {}
            tpr[att] = {}
            thresholds[att] = {}
            ece[att] = {}
            ks[att] = {}
            brier[att] = {}

            for j, subgroup in enumerate(subgroups[att]+['Global']):
                if approach == 'baseline':
                    r = collect_measures_baseline_or_fsn_or_ftc(
                        ground_truth['test'],
                        scores['test'],
                        confidences['test'],
                        nbins,
                        subgroup_scores['test'][att],
                        subgroup
                    )
                elif 'faircal' in approach:
                    r = collect_measures_bmc_or_oracle(
                        ground_truth['test'],
                        scores['test'],
                        confidences['test'],
                        nbins,
                        subgroup_scores['test'][att],
                        subgroup
                    )
                elif 'fsn' in approach:
                    r = collect_measures_baseline_or_fsn_or_ftc(
                        ground_truth['test'],
                        fair_scores['test'],
                        confidences['test'],
                        nbins,
                        subgroup_scores['test'][att],
                        subgroup
                    )
                elif approach == 'ftc':
                    r = collect_measures_baseline_or_fsn_or_ftc(
                        ground_truth['test'],
                        fair_scores['test'],
                        confidences['test'],
                        nbins,
                        subgroup_scores['test'][att],
                        subgroup
                    )
                elif approach == 'agenda':
                    r = collect_measures_baseline_or_fsn_or_ftc(
                        ground_truth['test'],
                        fair_scores['test'],
                        confidences['test'],
                        nbins,
                        subgroup_scores['test'][att],
                        subgroup
                    )
                elif approach == 'oracle':
                    r = collect_measures_bmc_or_oracle(
                        ground_truth['test'],
                        scores['test'],
                        confidences['test'][att],
                        nbins,
                        subgroup_scores['test'][att],
                        subgroup
                    )
                else:
                    raise ValueError('Approach %s not available.' % approach)

                fpr[att][subgroup] = r[0]
                tpr[att][subgroup] = r[1]
                thresholds[att][subgroup] = r[2]
                ece[att][subgroup] = r[3]
                ks[att][subgroup] = r[4]
                brier[att][subgroup] = r[5]

        data['fold' + str(fold)] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'ece': ece,
            'ks': ks,
            'brier': brier
        }


    return data


parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', type=str, nargs='+',
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
    help='approaches to use seperate using " "',
    choices=['baseline', 'faircal', 'fsn', 'agenda', 'ftc', 'oracle', 'all'],
    default='all')

parser.add_argument(
    '--calibration_methods', type=str, nargs='+',
    help='calibration_methods to use, seperate using " "',
    choices=['binning', 'isotonic_regression', 'beta', 'all'],
    default=['beta'])


def main():
    total_start = time.time()
    args = parser.parse_args()
    db = None

    for dataset in args.dataset:
        if dataset == 'rfw':
            db = pd.read_csv('data/rfw/rfw_w_sims.csv')
            nbins = 10
        elif dataset == 'bfw':
            db = pd.read_csv('data/bfw/bfw_w_sims.csv')
            nbins = 25
        else:
            print(f'Invalid dataset selected: {dataset}')
            return

        # Print out how many comparisons are possible for each model
        print(dataset)

        create_folder(f"{experiments_folder}/{dataset}")

        if args.features == 'all':
            if dataset == 'rfw':
                features = ['facenet', 'facenet-webface']
            else:
                features = ['facenet-webface', 'arcface']
        else:
            features = args.features

        if args.approaches == 'all':
            approaches = ['baseline', 'faircal', 'fsn', 'agenda', 'ftc', 'oracle']
        else:
            approaches = args.approaches

        if args.calibration_methods == 'all':
            calibration_methods = ['binning', 'isotonic_regression', 'beta']
        else:
            calibration_methods = args.calibration_methods

        print(dataset, features, approaches, calibration_methods)
        n_clusters = [100] #[500, 250, 150, 100, 75, 50, 25, 20, 15, 10, 5, 1] #n_clusters = 100 was used in the tables on page 8
        n_clusters = [4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        fpr_thr_list = [1e-3]
        for n_cluster in n_clusters:
            for fpr_thr in fpr_thr_list:
                print('fpr_thr: %.0e' % fpr_thr)
                for feature in features:
                    create_folder(f"{experiments_folder}/{dataset}/{feature}")
                    print('Feature: %s' % feature)
                    for approach in approaches:
                        create_folder(f"{experiments_folder}/{dataset}/{feature}/{approach}")
                        print('   Approach: %s' % approach)
                        for calibration_method in calibration_methods:
                            create_folder(f"{experiments_folder}/{dataset}/{feature}/{approach}/{calibration_method}")
                            print('      Calibration Method: %s' % calibration_method)
                            if 'faircal' in approach:
                                print('         number clusters: %d' % n_cluster)
                            elif 'fsn' in approach:
                                print('         number clusters: %d' % n_cluster)
                            saveto = file_name_save(dataset, feature, approach, calibration_method, nbins, n_cluster,
                                                    fpr_thr)
                            if os.path.exists(saveto):
                                os.remove(saveto)
                            prepare_dir(saveto)
                            np.save(saveto, {})

                            # Load embedding data and preprocess
                            embedding_data = pickle.load(open(f'embeddings/{feature}_{dataset}_embeddings.pk', 'rb'))
                            if dataset == 'bfw':
                                embedding_data['img_path'] = embedding_data['img_path'].apply(lambda x: x.replace('data/bfw/bfw-cropped-aligned/', ''))
                            if dataset == 'rfw':
                                embedding_data['img_path'] = embedding_data['img_path'].apply(lambda x: x.replace('data/rfw/data/', ''))
                            start = time.time()
                            data = gather_results(
                                dataset,
                                db,
                                nbins,
                                n_cluster,
                                fpr_thr,
                                feature,
                                approach,
                                calibration_method,
                                embedding_data
                            )
                            print(f'Dataset {dataset}, feature {feature}, approach {approach}: {round(time.time()-start, 3)} seconds')
                            np.save(saveto, data)
    print(f'All experiments took {round(time.time() - total_start)} seconds')



def collect_measures_bmc_or_oracle(ground_truth, scores, confidences, nbins, subgroup_scores, subgroup):
    if subgroup == 'Global':
        select = np.full(scores.size, True, dtype=bool)
    else:
        select = np.logical_and(subgroup_scores['left'] == subgroup, subgroup_scores['right'] == subgroup)

    r = roc_curve(ground_truth[select].astype(bool), confidences[select], drop_intermediate=False)
    fpr = {'calibration': r[0]}
    tpr = {'calibration': r[1]}
    thresholds = {'calibration': r[2]}
    ece, ks, brier = compute_scores(confidences[select], ground_truth[select], nbins)
    return fpr, tpr, thresholds, ece, ks, brier


def collect_measures_baseline_or_fsn_or_ftc(ground_truth, scores, confidences, nbins, subgroup_scores, subgroup):
    if subgroup == 'Global':
        select = np.full(scores.size, True, dtype=bool)
    else:
        select = np.logical_and(subgroup_scores['left'] == subgroup, subgroup_scores['right'] == subgroup)
    r = roc_curve(ground_truth[select].astype(bool), confidences[select], drop_intermediate=False)
    fpr = {'calibration': r[0]}
    tpr = {'calibration': r[1]}
    thresholds = {'calibration': r[2]}
    r = roc_curve(ground_truth[select].astype(bool), scores[select], drop_intermediate=False)
    fpr['pre_calibration'] = r[0]
    tpr['pre_calibration'] = r[1]
    thresholds['pre_calibration'] = r[2]
    ece, ks, brier = compute_scores(confidences[select], ground_truth[select], nbins)
    return fpr, tpr, thresholds, ece, ks, brier


def file_name_save(dataset, feature, approach, calibration_method, nbins, n_cluster, fpr_thr):
    folder_name = '/'.join([dataset, feature, approach, calibration_method])
    if 'faircal' in approach:
        file_name = '_'.join(['nbins', str(nbins), 'nclusters', str(n_cluster)])
    elif 'fsn' in approach:
        file_name = '_'.join(['nbins', str(nbins), 'nclusters', str(n_cluster), 'fpr', format(fpr_thr, '.0e')])
    else:
        file_name = '_'.join(['nbins', str(nbins)])
    saveto = f"{experiments_folder}{folder_name}/{file_name}.npy"
    return saveto


def create_folder(path):
    try:
        path = path.replace('//', '/')
        os.mkdir(path)
    except OSError as e:
        return None

if __name__ == '__main__':
    main()
