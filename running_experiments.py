import pickle
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import roc_curve

from approaches import ApproachManager
from utils import compute_scores


class MeasureCollecter:
    def __init__(self):
        self.nbins = None
        self.approach = None

    def collect_measures(self, approach, ground_truth, scores, fair_scores, confidences, subgroup, subgroup_scores, att):
        if approach in ('baseline', 'faircal', 'agenda', 'gmm-discrete'):
            score_to_assess = scores['test']
        elif approach in ('fsn', 'ftc', 'oracle'):
            score_to_assess = fair_scores['test']
        else:
            raise ValueError('Approach %s not available.' % self.approach)


        if approach in ('baseline', 'fsn', 'ftc', 'agenda', 'gmm-discrete'):
            collect_measures_func = self.collect_measures_baseline_or_fsn_or_ftc
        elif approach in ('faircal', 'oracle'):
            collect_measures_func = self.collect_measures_bmc_or_oracle
        else:
            raise ValueError('Approach %s not available.' % self.approach)

        if approach == 'oracle':
            confidences_to_assess = confidences['test'][att]
        else:
            confidences_to_assess = confidences['test']

        return collect_measures_func(
            ground_truth['test'], score_to_assess, confidences_to_assess, subgroup_scores['test'][att], subgroup
        )

    def collect_measures_bmc_or_oracle(self, ground_truth, scores, confidences, subgroup_scores, subgroup):
        if subgroup == 'Global':
            select = np.full(scores.size, True, dtype=bool)
        else:
            select = np.logical_and(subgroup_scores['left'] == subgroup, subgroup_scores['right'] == subgroup)

        r = roc_curve(ground_truth[select].astype(bool), confidences[select], drop_intermediate=False)
        fpr = {'calibration': r[0]}
        tpr = {'calibration': r[1]}
        thresholds = {'calibration': r[2]}
        ece, ks, brier = compute_scores(confidences[select], ground_truth[select], self.nbins)
        return fpr, tpr, thresholds, ece, ks, brier


    def collect_measures_baseline_or_fsn_or_ftc(self, ground_truth, scores, confidences, subgroup_scores, subgroup):
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
        ece, ks, brier = compute_scores(confidences[select], ground_truth[select], self.nbins)
        return fpr, tpr, thresholds, ece, ks, brier

class ExperimentRunner(ApproachManager, MeasureCollecter):
    def __init__(self):
        super().__init__()
        self.nbins = None
        self.subgroups = None
        self.sensitive_attributes = None

    def load_similarity_data(self):
        """
        Placeholder method to be overwritten.
        """
        return pd.DataFrame()

    def load_embedding_data(self):
        """
        Placeholder method to be overwritten.
        """
        return pd.DataFrame()

    def clean_sim_data(self, db, embedding_data):
        """
        Placeholder method to be overwritten.
        """
        return pd.DataFrame()

    def initialisation(self, db, fold):
        db_fold = {'cal': db[db['fold'] != fold], 'test': db[db['fold'] == fold]}
        scores = {}
        ground_truth = {}
        subgroup_scores = {}
        for dataset in ['cal', 'test']:
            scores[dataset] = np.array(db_fold[dataset][self.feature])
            ground_truth[dataset] = np.array(db_fold[dataset]['same'])
            subgroup_scores[dataset] = {}
            for att in self.subgroups.keys():
                subgroup_scores[dataset][att] = {}
                subgroup_scores[dataset][att]['left'] = np.array(db_fold[dataset][self.sensitive_attributes[att][0]])
                subgroup_scores[dataset][att]['right'] = np.array(db_fold[dataset][self.sensitive_attributes[att][1]])
        return db_fold, scores, ground_truth, subgroup_scores

    def run_experiment(self):
        db = self.load_similarity_data()
        embedding_data = self.load_embedding_data()
        db = self.clean_sim_data(db, embedding_data)
        experiment_output = {}
        for fold in range(1, 6):
            db_fold, scores, ground_truth, subgroup_scores = self.initialisation(db, fold)
            scores, ground_truth, confidences, fair_scores = self.get_metrics(
                embedding_data, db, db_fold, fold, scores, ground_truth, subgroup_scores
            )
            fpr, tpr, thresholds, ece, ks, brier = self.get_summary_stats(
                ground_truth, scores, fair_scores, confidences, subgroup_scores
            )
            experiment_output[f'fold{fold}'] = {
                'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'ece': ece, 'ks': ks, 'brier': brier
            }
        return experiment_output

    def get_summary_stats(self, ground_truth, scores, fair_scores, confidences, subgroup_scores):
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
            for j, subgroup in enumerate(self.subgroups[att] + ['Global']):
                r = self.collect_measures(
                    self.approach, ground_truth, scores, fair_scores, confidences, subgroup, subgroup_scores, att
                )
                fpr[att][subgroup] = r[0]
                tpr[att][subgroup] = r[1]
                thresholds[att][subgroup] = r[2]
                ece[att][subgroup] = r[3]
                ks[att][subgroup] = r[4]
                brier[att][subgroup] = r[5]
        return fpr, tpr, thresholds, ece, ks, brier


class RfwExperimentRunner(ExperimentRunner):
    all_features = ['facenet', 'facenet-webface']
    def __init__(self, n_cluster, fpr_thr, feature, approach, calibration_method):
        super().__init__()
        self.n_cluster = n_cluster
        self.fpr_thr = fpr_thr
        self.feature = feature
        self.approach = approach
        self.calibration_method = calibration_method

        # RFW specific variables
        self.dataset = 'rfw'
        self.nbins = 10
        self.subgroups = {'ethnicity': ['African', 'Asian', 'Caucasian', 'Indian']}
        self.sensitive_attributes = {'ethnicity': ['ethnicity', 'ethnicity']}

    def load_embedding_data(self):
        embedding_data = pickle.load(open(f'embeddings/{self.feature}_rfw_embeddings.pk', 'rb'))
        embedding_data['img_path'] = embedding_data['img_path'].map(lambda x: x.replace('data/rfw/data/', ''))
        return embedding_data

    def load_similarity_data(self):
        return pd.read_csv('data/rfw/rfw_w_sims.csv')

    def clean_sim_data(self, db, embedding_data):
        """
        Placeholder method to be overwritten.
        """
        # Assigning the embedding to each image
        db['image_id_1_clean'] = db['id1'].map(str) + '_000' + db['num1'].map(str)
        db['image_id_2_clean'] = db['id2'].map(str) + '_000' + db['num2'].map(str)
        embedding_map = dict(zip(embedding_data['image_id'], embedding_data['embedding']))
        db['emb_1'] = db['image_id_1_clean'].map(embedding_map)
        db['emb_2'] = db['image_id_2_clean'].map(embedding_map)
        # Create a mask to remove the rows with missing cosine similarity or embedding.
        keep_cond = (
            db[self.feature].notnull() &
            db['emb_1'].notnull() &
            db['emb_2'].notnull()
        )
        return db[keep_cond].reset_index(drop=True)

    def collect_miscellania(self, n_clusters, cluster_model, db_fold, embedding_data):
        # setup clusters
        clusters = {}
        for i_cluster in range(n_clusters):
            clusters[i_cluster] = {}

            for variable in ['scores', 'ground_truth']:
                clusters[i_cluster][variable] = {}
                for dataset in ['cal', 'test']:
                    clusters[i_cluster][variable][dataset] = []
        scores = {}
        ground_truth = {}
        cluster_scores = {}
        for dataset in ['cal', 'test']:
            scores[dataset] = np.zeros(len(db_fold[dataset]))
            ground_truth[dataset] = np.zeros(len(db_fold[dataset])).astype(bool)
            cluster_scores[dataset] = np.zeros((len(db_fold[dataset]), 2)).astype(int)

        # collect scores, ground_truth per cluster for the calibration set

        # Predict kmeans
        embedding_data['i_cluster'] = cluster_model.predict(
            np.vstack(embedding_data['embedding'].to_numpy()).astype('float32'))
        cluster_map = dict(zip(embedding_data['img_path'], embedding_data['i_cluster']))

        for dataset, db in zip(['cal', 'test'], [db_fold['cal'], db_fold['test']]):
            scores[dataset] = np.array(db[self.feature])
            ground_truth[dataset] = np.array(db['same'].astype(bool))

            db['path1'] = db['ethnicity'] + '/' + db['id1'] + '/' + db['id1'] + '_000' + db['num1'].map(
                str) + '.jpg'
            db['path2'] = db['ethnicity'] + '/' + db['id2'] + '/' + db['id2'] + '_000' + db['num2'].map(
                str) + '.jpg'

            db[f'{dataset}_cluster_1'] = db['path1'].map(cluster_map)
            db[f'{dataset}_cluster_2'] = db['path2'].map(cluster_map)

            if db[[f'{dataset}_cluster_1', f'{dataset}_cluster_2']].isnull().sum().sum():
                print('Warning: There should not be nans in the cluster columns.')

            cluster_scores[dataset] = db[[f'{dataset}_cluster_1', f'{dataset}_cluster_2']].values

        return scores, ground_truth, clusters, cluster_scores

    def collect_embeddings(self, db_cal, embedding_data):
        # Collect embeddings of all the images in the calibration set
        all_embeddings = np.empty((0, 512))
        embedding_data['embedding'] = embedding_data['embedding'].to_list()

        # Iterating over subgroup necessary for creating correct image_path string
        for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
            select = db_cal[db_cal['ethnicity'] == subgroup]
            if select.empty:
                continue
            select['num1'] = select['num1'].astype('string')
            select['path1'] = subgroup + '/' + select['id1'] + '/' + select['id1'] + '_000' + select['num1'] + '.jpg'
            select['num2'] = select['num2'].astype('string')
            select['path2'] = subgroup + '/' + select['id2'] + '/' + select['id2'] + '_000' + select['num2'] + '.jpg'

            file_names = set(select['path1'].tolist()) | set(select['path2'].tolist())
            embeddings = embedding_data[embedding_data['img_path'].isin(file_names)]['embedding'].to_numpy()
            embeddings = np.vstack(embeddings)
            all_embeddings = np.concatenate([all_embeddings, embeddings])

        return all_embeddings

    def collect_pair_embeddings(self, db_cal):
        embeddings = {
            'left': torch.tensor(np.vstack(db_cal['emb_1'].values)),
            'right': torch.tensor(np.vstack(db_cal['emb_2'].values))
        }
        ground_truth = np.array(db_cal['same'].astype(bool))
        subgroups = np.array(db_cal['ethnicity'])
        return embeddings, ground_truth, subgroups, subgroups

    def collect_embeddings_rfw(self, db_cal, embedding_data):
        all_images = set(db_cal['image_id_1_clean']) | set(db_cal['image_id_2_clean'])
        df_all_images = pd.DataFrame([{'image_id': p} for p in all_images])
        embedding_map = dict(zip(embedding_data['image_id'], embedding_data['embedding']))

        subgroup_map = {
            **dict(zip(db_cal['image_id_1_clean'], db_cal['ethnicity'])),
            **dict(zip(db_cal['image_id_2_clean'], db_cal['ethnicity'])),
        }
        id_map = {
            **dict(zip(db_cal['image_id_1_clean'], db_cal['id1'])),
            **dict(zip(db_cal['image_id_2_clean'], db_cal['id2'])),
        }
        df_all_images['embedding'] = df_all_images['image_id'].map(embedding_map)
        df_all_images = df_all_images[df_all_images['embedding'].notnull()].reset_index(drop=True)

        df_all_images['att'] = df_all_images['image_id'].map(subgroup_map)
        df_all_images['id'] = df_all_images['image_id'].map(id_map)

        embeddings = np.vstack(df_all_images['embedding'].to_numpy())
        subgroup_embeddings = pd.Series(df_all_images['att'], dtype="category").cat.codes.values
        id_embeddings = df_all_images['id']

        return embeddings, subgroup_embeddings, id_embeddings

    def collect_embeddings_agenda(self, db_cal, embedding_data):
        all_images = set(db_cal['image_id_1_clean']) | set(db_cal['image_id_2_clean'])
        df_all_images = pd.DataFrame([{'image_id': p} for p in all_images])
        embedding_map = dict(zip(embedding_data['image_id'], embedding_data['embedding']))

        subgroup_map = {
            **dict(zip(db_cal['image_id_1_clean'], db_cal['ethnicity'])),
            **dict(zip(db_cal['image_id_2_clean'], db_cal['ethnicity'])),
        }
        id_map = {
            **dict(zip(db_cal['image_id_1_clean'], db_cal['id1'])),
            **dict(zip(db_cal['image_id_2_clean'], db_cal['id2'])),
        }
        df_all_images['embedding'] = df_all_images['image_id'].map(embedding_map)
        df_all_images = df_all_images[df_all_images['embedding'].notnull()].reset_index(drop=True)

        df_all_images['att'] = df_all_images['image_id'].map(subgroup_map)
        df_all_images['id'] = df_all_images['image_id'].map(id_map)

        embeddings = np.vstack(df_all_images['embedding'].to_numpy())
        subgroup_embeddings = pd.Series(df_all_images['att'], dtype="category").cat.codes.values
        id_embeddings = df_all_images['id']

        return embeddings, subgroup_embeddings, id_embeddings

    def collect_error_embeddings(self, db_cal):
        embeddings = {
            'left': torch.tensor(np.vstack(db_cal['emb_1'].values)),
            'right': torch.tensor(np.vstack(db_cal['emb_2'].values))
        }
        ground_truth = np.array(db_cal['same']).astype(bool)
        subgroups = np.array(db_cal['ethnicity'])
        error_embeddings = torch.abs(embeddings['left'] - embeddings['right'])

        return error_embeddings, ground_truth, subgroups, subgroups

class BfwExperimentRunner(ExperimentRunner):
    all_features = ['facenet-webface', 'arcface']
    def __init__(self, n_cluster, fpr_thr, feature, approach, calibration_method):
        super().__init__()
        self.n_cluster = n_cluster
        self.fpr_thr = fpr_thr
        self.feature = feature
        self.approach = approach
        self.calibration_method = calibration_method

        # RFW specific variables
        self.dataset = 'bfw'
        self.nbins = 25
        self.subgroups = {
            'e': ['B', 'A', 'W', 'I'],
            'g': ['F', 'M'],
            'att': ['asian_females', 'asian_males', 'black_males', 'black_females', 'white_females', 'white_males',
                    'indian_females', 'indian_males']
        }
        self.sensitive_attributes = {'e': ['e1', 'e2'], 'g': ['g1', 'g2'], 'att': ['att1', 'att2']}

    def load_embedding_data(self):
        embedding_data = pickle.load(open(f'embeddings/{self.feature}_bfw_embeddings.pk', 'rb'))
        embedding_data['img_path'] = embedding_data['img_path'].map(
            lambda x: x.replace('data/bfw/bfw-cropped-aligned/', '')
        )
        return embedding_data

    def load_similarity_data(self):
        return pd.read_csv('data/bfw/bfw_w_sims.csv')

    def clean_sim_data(self, db, embedding_data):
        """
        Placeholder method to be overwritten.
        """
        # Assigning the embedding to each image
        embedding_map = dict(zip(embedding_data['img_path'], embedding_data['embedding']))
        db['emb_1'] = db['path1'].map(embedding_map)
        db['emb_2'] = db['path2'].map(embedding_map)

        # Create a mask to remove the rows with missing cosine similarity or embedding.
        keep_cond = (
                db[self.feature].notnull() &
                db['emb_1'].notnull() &
                db['emb_2'].notnull()
        )
        return db[keep_cond]

    def collect_miscellania(self, n_clusters, kmeans, db_fold, embedding_data):
        # setup clusters
        clusters = {}
        for i_cluster in range(n_clusters):
            clusters[i_cluster] = {}

            for variable in ['scores', 'ground_truth']:
                clusters[i_cluster][variable] = {}
                for dataset in ['cal', 'test']:
                    clusters[i_cluster][variable][dataset] = []
        scores = {}
        ground_truth = {}
        cluster_scores = {}
        for dataset in ['cal', 'test']:
            # consider only pairs that have cosine similarities
            number_pairs = len(db_fold[dataset][db_fold[dataset][self.feature].notna()])
            scores[dataset] = np.zeros(number_pairs)
            ground_truth[dataset] = np.zeros(number_pairs).astype(bool)
            cluster_scores[dataset] = np.zeros((number_pairs, 2)).astype(int)

        # Predict kmeans
        embedding_data['i_cluster'] = kmeans.predict(np.vstack(embedding_data['embedding'].to_numpy()))
        cluster_map = dict(zip(embedding_data['img_path'], embedding_data['i_cluster']))

        # Collect cluster info for each pair of images
        for dataset, db in zip(['cal', 'test'], [db_fold['cal'], db_fold['test']]):
            # remove image pairs that have missing cosine similarities
            db = db[db[self.feature].notna()].reset_index(drop=True)
            scores[dataset] = np.array(db[self.feature])
            ground_truth[dataset] = np.array(db['same'].astype(bool))

            db[f'{dataset}_cluster_1'] = db['path1'].map(cluster_map)
            db[f'{dataset}_cluster_2'] = db['path2'].map(cluster_map)

            if db[[f'{dataset}_cluster_1', f'{dataset}_cluster_2']].isnull().sum().sum():
                print('Warning: There should not be nans in the cluster columns.')
            cluster_scores[dataset] = db[[f'{dataset}_cluster_1', f'{dataset}_cluster_2']].values
            print(f'{dataset}: {cluster_scores[dataset].shape}')

        return scores, ground_truth, clusters, cluster_scores

    def collect_embeddings(self, db_cal, embedding_data):
        # Collect embeddings of all the images in the calibration set
        embedding_data['embedding'] = embedding_data['embedding'].to_list()
        file_names = set(db_cal['path1'].tolist()) | set(db_cal['path2'].tolist())
        embeddings = embedding_data[embedding_data['img_path'].isin(file_names)]['embedding'].to_numpy()
        embeddings = np.vstack(embeddings)
        return embeddings

    def collect_pair_embeddings(self, db_cal):
        # collect embeddings of all the images in the calibration set
        embeddings = {
            'left': torch.from_numpy(np.vstack(db_cal['emb_1'].values)),
            'right': torch.from_numpy(np.vstack(db_cal['emb_2'].values)),
        }
        ground_truth = np.array(db_cal['same'].astype(bool))
        subgroups_left = np.array(db_cal['att1'])
        subgroups_right = np.array(db_cal['att2'])
        return embeddings, ground_truth, subgroups_left, subgroups_right

    def collect_embeddings_agenda(self, db_cal, embedding_data):
        all_images = set(db_cal['path1']) | set(db_cal['path2'])
        df_all_images = pd.DataFrame([{'img_path': p} for p in all_images])
        embedding_map = dict(zip(embedding_data['img_path'], embedding_data['embedding']))

        subgroup_map = {
            **dict(zip(db_cal['path1'], db_cal['att1'])),
            **dict(zip(db_cal['path2'], db_cal['att2'])),
        }
        id_map = {
            **dict(zip(db_cal['path1'], db_cal['id1'])),
            **dict(zip(db_cal['path2'], db_cal['id2'])),
        }
        df_all_images['embedding'] = df_all_images['img_path'].map(embedding_map)
        df_all_images = df_all_images[df_all_images['embedding'].notnull()].reset_index(drop=True)

        df_all_images['att'] = df_all_images['img_path'].map(subgroup_map)
        df_all_images['id'] = df_all_images['img_path'].map(id_map)

        embeddings = np.vstack(df_all_images['embedding'].to_numpy())
        subgroup_embeddings = pd.Series(df_all_images['att'], dtype="category").cat.codes.values
        id_embeddings = df_all_images['img_path']

        return embeddings, subgroup_embeddings, id_embeddings

    def collect_error_embeddings(self, db_cal):
        # collect embeddings of all the images in the calibration set
        embeddings = {
            'left': torch.tensor(np.vstack(db_cal['emb_1'].values)),
            'right': torch.tensor(np.vstack(db_cal['emb_2'].values))
        }
        ground_truth = np.array(db_cal['same']).astype(bool)
        subgroups_left = np.array(db_cal['att1'])
        subgroups_right = np.array(db_cal['att2'])
        error_embeddings = torch.abs(embeddings['left'] - embeddings['right'])
        return error_embeddings, ground_truth, subgroups_left, subgroups_right



