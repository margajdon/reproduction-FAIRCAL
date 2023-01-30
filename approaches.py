import numpy as np
import os
import pandas as pd
from pycave.bayes import GaussianMixture
from pycave.clustering import KMeans

from approaches_agenda import AgendaApproach
from approaches_ftc import FtcApproach

pd.options.mode.chained_assignment = None

from sklearn.metrics import roc_curve

from calibration_methods import BinningCalibration
from calibration_methods import IsotonicCalibration
from calibration_methods import BetaCalibration
from utils import prepare_dir

import torch

experiments_folder = 'experiments/'


class ApproachManager(AgendaApproach, FtcApproach):
    nbins = None
    dataset = None
    approach = None
    calibration_method = None
    feature = None
    n_cluster = None
    fpr_thr = None
    subgroups = None

    def get_calibration_method(self, scores, ground_truth, score_min=-1, score_max=1):
        if self.calibration_method == 'binning':
            calibration = BinningCalibration(
                scores['cal'], ground_truth['cal'], score_min=score_min, score_max=score_max, nbins=self.nbins
            )
        elif self.calibration_method == 'isotonic_regression':
            calibration = IsotonicCalibration(
                scores['cal'], ground_truth['cal'], score_min=score_min, score_max=score_max
            )
        elif self.calibration_method == 'beta':
            calibration = BetaCalibration(scores['cal'], ground_truth['cal'], score_min=score_min, score_max=score_max)
        else:
            raise ValueError('Calibration method %s not available' % self.calibration_method)
        return calibration

    def baseline(self, scores, ground_truth, score_min=-1, score_max=1):
        calibration = self.get_calibration_method(scores, ground_truth, score_min, score_max)
        return {'cal': calibration.predict(scores['cal']), 'test': calibration.predict(scores['test'])}

    def oracle(self, scores, ground_truth, subgroup_scores):
        confidences = {}
        for dataset in ['cal', 'test']:
            confidences[dataset] = {}
            for att in self.subgroups.keys():
                confidences[dataset][att] = np.zeros(len(scores[dataset]))

        for att in self.subgroups.keys():
            for subgroup in self.subgroups[att]:
                select = {}
                for dataset in ['cal', 'test']:
                    select[dataset] = np.logical_and(
                        subgroup_scores[dataset][att]['left'] == subgroup,
                        subgroup_scores[dataset][att]['right'] == subgroup
                    )
                scores_cal_subgroup = scores['cal'][select['cal']]
                ground_truth_cal_subgroup = ground_truth['cal'][select['cal']]
                calibration = self.get_calibration_method(scores_cal_subgroup, ground_truth_cal_subgroup)
                confidences['cal'][att][select['cal']] = calibration.predict(scores_cal_subgroup)
                confidences['test'][att][select['test']] = calibration.predict(scores['test'][select['test']])
        return confidences

    def cluster_methods(self, fold, db_fold, score_normalization, fpr, embedding_data, km_random_state=42):
        # k-means algorithm
        saveto = f"experiments/kmeans/{self.dataset}_{self.feature}_nclusters{self.n_cluster}_fold{fold}.npy"
        if os.path.exists(saveto):
            os.remove(saveto)
        prepare_dir(saveto)
        np.save(saveto, {})
        embeddings = self.collect_embeddings(db_fold['cal'], embedding_data)

        cluster_method = None
        gpu_bool = torch.cuda.is_available()
        if self.approach in ('faircal', 'fsn'):
            if gpu_bool:
                cluster_method = KMeans(num_clusters=self.n_cluster, trainer_params=dict(accelerator='gpu', devices=1))
            else:
                cluster_method = KMeans(num_clusters=self.n_cluster)
        elif self.approach == 'gmm-discrete':
            if gpu_bool:
                cluster_method = GaussianMixture(
                    num_components=self.n_cluster, trainer_params=dict(accelerator='gpu', devices=1)
                )
            else:
                cluster_method = GaussianMixture(num_components=self.n_cluster)

        else:
            raise ValueError(f"Approach {self.approach} does not map to a clustering algorithm!")

        gpu_bool = torch.cuda.is_available()

        cluster_method.fit(embeddings.astype('float32'))
        np.save(saveto, cluster_method)

        r = self.collect_miscellania(self.n_cluster, cluster_method, db_fold, embedding_data)

        scores, ground_truth, clusters, cluster_scores = r[:4]

        print('Statistics Cluster K = %d' % self.n_cluster)
        stats = np.zeros(self.n_cluster)
        for i_cluster in range(self.n_cluster):
            select = np.logical_or(cluster_scores['cal'][:, 0] == i_cluster, cluster_scores['cal'][:, 1] == i_cluster)
            clusters[i_cluster]['scores']['cal'] = scores['cal'][select]
            clusters[i_cluster]['ground_truth']['cal'] = ground_truth['cal'][select]
            stats[i_cluster] = len(clusters[i_cluster]['scores']['cal'])

        print('Minimum number of pairs in clusters %d' % (min(stats)))
        print('Maximum number of pairs in clusters %d' % (max(stats)))
        print('Median number of pairs in clusters %1.1f' % (np.median(stats)))
        print('Mean number of pairs in clusters %1.1f' % (np.mean(stats)))

        if score_normalization:

            global_threshold = self.find_threshold(scores['cal'], ground_truth['cal'], fpr)
            local_threshold = np.zeros(self.n_cluster)

            for i_cluster in range(self.n_cluster):
                scores_cal = clusters[i_cluster]['scores']['cal']
                ground_truth_cal = clusters[i_cluster]['ground_truth']['cal']
                local_threshold[i_cluster] = self.find_threshold(scores_cal, ground_truth_cal, fpr)

            fair_scores = {}

            for dataset in ['cal', 'test']:
                fair_scores[dataset] = np.zeros(len(scores[dataset]))
                for i_cluster in range(self.n_cluster):
                    for t in [0, 1]:
                        select = cluster_scores[dataset][:, t] == i_cluster
                        fair_scores[dataset][select] += local_threshold[i_cluster] - global_threshold

                fair_scores[dataset] = scores[dataset] - fair_scores[dataset] / 2

            # The fair scores are no longer cosine similarity scores so they may not lie in the interval [-1,1]
            fair_scores_max = 1 - min(local_threshold - global_threshold)
            fair_scores_min = -1 - max(local_threshold - global_threshold)

            confidences = self.baseline(
                fair_scores,
                ground_truth,
                score_min=fair_scores_min,
                score_max=fair_scores_max
            )
        else:
            fair_scores = {}
            confidences = {}

            # Fit calibration
            cluster_calibration_method = {}
            for i_cluster in range(self.n_cluster):
                scores_temp = clusters[i_cluster]['scores']
                ground_truth_temp = clusters[i_cluster]['ground_truth']
                cluster_calibration_method[i_cluster] = self.get_calibration_method(scores_temp, ground_truth_temp)
                clusters[i_cluster]['confidences'] = {
                    'cal': cluster_calibration_method[i_cluster].predict(scores_temp['cal'])
                }

            for dataset in ['cal', 'test']:
                confidences[dataset] = np.zeros(len(scores[dataset]))
                p = np.zeros(len(scores[dataset]))
                for i_cluster in range(self.n_cluster):
                    for t in [0, 1]:
                        select = cluster_scores[dataset][:, t] == i_cluster
                        aux = scores[dataset][select]
                        if len(aux) > 0:
                            aux = cluster_calibration_method[i_cluster].predict(aux)
                            confidences[dataset][select] += aux * stats[i_cluster]
                            p[select] += stats[i_cluster]
                confidences[dataset] = confidences[dataset] / p

        return scores, ground_truth, confidences, fair_scores

    def get_metrics(
            self, embedding_data, db, db_fold, fold, scores, ground_truth, subgroup_scores
    ):
        fair_scores = None
        if self.approach == 'baseline':
            confidences = self.baseline(scores, ground_truth)
        elif self.approach in ('faircal', 'gmm-discrete'):
            scores, ground_truth, confidences, fair_scores = self.cluster_methods(
                fold,
                db_fold,
                score_normalization=False,
                fpr=0,
                embedding_data=embedding_data
            )
        elif self.approach == 'fsn':
            scores, ground_truth, confidences, fair_scores = self.cluster_methods(
                fold,
                db_fold,
                score_normalization=True,
                fpr=self.fpr_thr,
                embedding_data=embedding_data
            )
        elif self.approach == 'ftc':
            fair_scores, confidences, model = self.ftc(
                db_fold
            )
            to_join = [self.dataset, self.calibration_method, self.feature, 'fold', str(fold)]
            saveto = 'experiments/ftc_settings/' + '_'.join(to_join)
            prepare_dir(saveto)
            torch.save(model.state_dict(), saveto)
        elif self.approach == 'agenda':
            fair_scores, confidences, modelM, modelC, modelE = self.agenda(
                db_fold, embedding_data
            )

            to_join = [self.dataset, self.calibration_method, self.feature, 'fold', str(fold)]
            saveto = 'experiments/agenda_settings/' + '_'.join(to_join)
            prepare_dir(saveto)
            torch.save(modelM.state_dict(), saveto + '_modelM')
            torch.save(modelC.state_dict(), saveto + '_modelC')
            torch.save(modelE.state_dict(), saveto + '_modelE')
        elif self.approach == 'oracle':
            confidences = self.oracle(
                scores, ground_truth, subgroup_scores
            )
        else:
            raise ValueError('Approach %s not available.' % self.approach)
        return scores, ground_truth, confidences, fair_scores

    def find_threshold(self, scores, ground_truth, fpr_threshold):
        far, tar, thresholds = roc_curve(ground_truth, scores, drop_intermediate=True)
        aux = np.abs(far - fpr_threshold)
        return np.min(thresholds[aux == np.min(aux)])

    def collect_embeddings(self, db_cal, embedding_data):
        """
        Placeholder method to be overwritten
        """
        return pd.DataFrame()

    def collect_miscellania(self, n_clusters, kmeans, db_fold, embedding_data):
        """
        Placeholder method to be overwritten
        """
        return {}, None, None, None




