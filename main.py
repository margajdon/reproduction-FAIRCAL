import itertools

import pandas as pd
import numpy as np
import os
import argparse
import torch
import pickle

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


class DataLoader:
    @staticmethod
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


class MeasuresCollector:
    @staticmethod
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

    @staticmethod
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

class Analyzer(DataLoader, MeasuresCollector):
    def __init__(self, dataset, features, calibration_methods, approaches):
        self.dataset = dataset
        self.features = features
        self.calibration_methods = calibration_methods
        self.approaches = approaches
        self.data_path, self.nbins, self.subgroups, self.sensitive_attributes = self.set_up(dataset)
        self.collection_methods = {
            'b_fsn_ftc': self.collect_measures_baseline_or_fsn_or_ftc,
            'bmc_oracel': self.collect_measures_bmc_or_oracle
        }

    def set_up(self, dataset):
        if dataset == 'rfw':
            data_path = 'data/rfw/rfw_w_sims.csv'
            nbins = 10
            subgroups = {'ethnicity': ['African', 'Asian', 'Caucasian', 'Indian']}
            sensitive_attributes = {'ethnicity': ['ethnicity', 'ethnicity']}
        elif dataset == 'bfw':
            data_path = 'data/bfw/bfw_w_sims.csv'
            nbins = 25
            subgroups = {
                'e': ['B', 'A', 'W', 'I'],
                'g': ['F', 'M'],
                'att': ['asian_females', 'asian_males', 'black_males', 'black_females', 'white_females', 'white_males',
                        'indian_females', 'indian_males']
            }
            sensitive_attributes = {'e': ['e1', 'e2'], 'g': ['g1', 'g2'], 'att': ['att1', 'att2']}
        else:
            raise ValueError(f'Unrecognised dataset {dataset}')

        return data_path, nbins, subgroups, sensitive_attributes

    def prep_data(self, db, dataset_name, embedding_data, feature):
        if dataset_name == 'rfw':
            db['image_id_1_clean'] = db['id1'].map(str) + '_000' + db['num1'].map(str)
            db['image_id_2_clean'] = db['id2'].map(str) + '_000' + db['num2'].map(str)
            embedding_map = dict(zip(embedding_data['image_id'], embedding_data['embedding']))
            db['emb_1'] = db['image_id_1_clean'].map(embedding_map)
            db['emb_2'] = db['image_id_2_clean'].map(embedding_map)
            keep_cond = (
                    db[feature].notna() &
                    db['emb_1'].notnull() &
                    db['emb_2'].notnull()
            )
        elif dataset_name == 'bfw':
            keep_cond = db[feature].notna()
        else:
            raise ValueError(f'Unrecognized dataset_name: {dataset_name}')

        # remove image pairs that have missing cosine similarities
        return db[keep_cond].reset_index(drop=True)

    def gather_results(self, db_input, embedding_data, n_cluster, fpr_thr, feature, approach, calibration_method):
        dataset_name = self.dataset
        nbins = self.nbins

        db = self.prep_data(db_input, dataset_name, embedding_data, feature)

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
                for att in self.subgroups.keys():
                    subgroup_scores[dataset][att] = {
                        'left': np.array(db_fold[dataset][self.sensitive_attributes[att][0]]),
                        'right': np.array(db_fold[dataset][self.sensitive_attributes[att][1]])
                    }

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
                    n_cluster,
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
                    n_cluster,
                    True,
                    fpr_thr,
                    embedding_data
                )
            elif approach == 'ftc':
                fair_scores, confidences, model = ftc(dataset_name, feature, db_fold, nbins, calibration_method)
                saveto = '_'.join(
                    [dataset_name, calibration_method, feature, 'fold', str(fold)])
                saveto = ftc_settings_folder + saveto
                torch.save(model.state_dict(), saveto)
            elif approach == 'agenda':
                fair_scores, confidences, modelM, modelC, modelE = agenda(
                    dataset_name, feature, db_fold, nbins, calibration_method, embedding_data
                )
                saveto = '_'.join(
                    [dataset_name, calibration_method, feature, 'fold', str(fold)])
                saveto = agenda_settings_folder + saveto
                torch.save(modelM.state_dict(), saveto+'_modelM')
                torch.save(modelC.state_dict(), saveto+'_modelC')
                torch.save(modelE.state_dict(), saveto+'_modelE')
            elif approach == 'oracle':
                confidences = oracle(scores, ground_truth, subgroup_scores, self.subgroups, nbins, calibration_method)
            else:
                raise ValueError('Approach %s not available.' % approach)


            fpr = {}
            tpr = {}
            thresholds = {}
            ece = {}
            ks = {}
            brier = {}

            for att in self.subgroups.keys():
                fpr[att] = {}
                tpr[att] = {}
                thresholds[att] = {}
                ece[att] = {}
                ks[att] = {}
                brier[att] = {}

                for j, subgroup in enumerate(self.subgroups[att]+['Global']):

                    if approach in ('baseline', 'faircal', 'oracle'):
                        score_to_report = scores['test']
                    elif approach in ('fsn', 'ftc', 'agenda'):
                        score_to_report = scores['test']
                    else:
                        raise ValueError('Approach %s not available.' % approach)

                    if approach in ('baseline', 'fsn', 'ftc', 'agenda'):
                        collection_method = 'b_fsn_ftc'
                    elif approach in ('faircal', 'oracle'):
                        collection_method = 'b_fsn_ftc'
                    else:
                        raise ValueError('Approach %s not available.' % approach)

                    value_dic = {
                        'ground_truth': ground_truth['test'],
                        'scores': score_to_report,
                        'confidences': confidences['test'],
                        'nbins': nbins,
                        'subgroup_scores': subgroup_scores['test'][att],
                        'subgroup': subgroup
                    }
                    r = self.collection_methods[collection_method](**value_dic)
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

    def main(self):
        dataset = self.dataset
        db = pd.read_csv(self.data_path)
        if args.features == 'all':
            if args.dataset == 'rfw':
                features = ['facenet', 'facenet-webface']
            else:
                features = ['facenet-webface', 'arcface']
        else:
            features = [args.features]
        if args.approaches == 'all':
            approaches = ['baseline', 'faircal', 'fsn', 'agenda', 'ftc', 'oracle']
        else:
            approaches = [args.approaches]
        if args.calibration_methods == 'all':
            calibration_methods = ['binning', 'isotonic_regression', 'beta']
        else:
            calibration_methods = [args.calibration_methods]
        n_clusters = [100] #[500, 250, 150, 100, 75, 50, 25, 20, 15, 10, 5, 1] #n_clusters = 100 was used in the tables on page 8
        fpr_thr_list = [1e-3]

        for to_unpack in itertools.product(n_clusters, fpr_thr_list, features, approaches, calibration_methods):
            n_cluster, fpr_thr, feature, approach, calibration_method = to_unpack
            print('fpr_thr: %.0e' % fpr_thr)
            print('Feature: %s' % feature)
            print('   Approach: %s' % approach)
            print('   Approach: %s' % approach)
            print('      Calibration Method: %s' % calibration_method)
            if approach in ('faircal', 'fsn'):
                print('         number clusters: %d' % n_cluster)

            saveto = self.file_name_save(
                dataset, feature, approach, calibration_method, self.nbins, n_cluster, fpr_thr
            )
            if os.path.exists(saveto):
                os.remove(saveto)
            prepare_dir(saveto)
            np.save(saveto, {})

            # Load embedding data and preprocess
            embedding_data = pickle.load(open(f'embeddings/{feature}_{dataset}_embeddings.pk', 'rb'))
            if dataset == 'bfw':
                embedding_data['img_path'] = embedding_data['img_path'].apply(
                    lambda x: x.replace('data/bfw/bfw-cropped-aligned/', ''))
            if dataset == 'rfw':
                embedding_data['img_path'] = embedding_data['img_path'].apply(lambda x: x.replace('data/rfw/data/', ''))
            data = self.gather_results(
                db, embedding_data, n_cluster, fpr_thr, feature, approach, calibration_method
            )
            np.save(saveto, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str,
        help='name of dataset',
        choices=['rfw', 'bfw'],
        default='bfw')

    parser.add_argument(
        '--features', type=str,
        help='features',
        choices=['facenet', 'facenet-webface', 'arcface', 'all'],
        default='all')

    parser.add_argument(
        '--approaches', type=str,
        help='approaches',
        choices=['baseline', 'faircal', 'fsn', 'agenda', 'ftc', 'oracle', 'all'],
        default='all')

    parser.add_argument(
        '--calibration_methods', type=str,
        help='calibration_methods',
        choices=['binning', 'isotonic_regression', 'beta', 'all'],
        default='all')

    args = parser.parse_args()
    args.calibration_methods = 'binning'
    args.features = 'facenet-webface'
    args.dataset = 'bfw'
    args.approaches = 'faircal'

    analyzer = Analyzer(args.dataset, args.features, args.calibration_methods, args.approaches)
    analyzer.main()
