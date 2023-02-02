import numpy as np
import os
import pandas as pd
from pycave.bayes import GaussianMixture
from pycave.clustering import KMeans
from sklearn.mixture import GaussianMixture as SkGaussianMixture
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

pd.options.mode.chained_assignment = None

from sklearn.metrics import roc_curve
from calibration_methods import BinningCalibration
from calibration_methods import IsotonicCalibration
from calibration_methods import BetaCalibration
from utils import prepare_dir, set_seed


experiments_folder = 'experiments/'


class AgendaApproach:
    """
    The AgendaApproach class contains all the methods that are specific to the Agenda approach.
    """
    dataset = None
    def agenda(self, db_fold, embedding_data):

        embeddings, subgroup_embeddings, id_embeddings = self.collect_embeddings_agenda(db_fold['cal'], embedding_data)

        embeddings_train, embeddings_test, id_train, id_test, subgroup_train, subgroup_test \
            = train_test_split(embeddings, id_embeddings, subgroup_embeddings, test_size=0.2)

        id_train = pd.Series(id_train, dtype="category").cat.codes.values
        id_test = pd.Series(id_test, dtype="category").cat.codes.values

        train_dataloader = DataLoader(
            AgendaEmbeddingsDataset(embeddings_train, id_train, subgroup_train),
            batch_size=400,
            shuffle=True,
            num_workers=0
        )
        test_dataloader = DataLoader(
            AgendaEmbeddingsDataset(embeddings_test, id_test, subgroup_test),
            batch_size=400,
            shuffle=True,
            num_workers=0
        )

        n_id = len(np.unique(id_train))
        n_subgroup = len(np.unique(subgroup_train))


        Nep = 100
        Tep = 10
        epochs_stage1 = 50
        epochs_stage2 = 25
        epochs_stage3 = 5
        epochs_stage4 = 5

        loss_fn = nn.CrossEntropyLoss()
        # Initialize
        modelM = NeuralNetworkM().cuda()
        modelC = NeuralNetworkC(n_id).cuda()

        # Set the optimizer
        optimizer_stage1 = optim.Adam(list(modelM.parameters())+list(modelC.parameters()), lr=1e-3)
        # Set the seed
        set_seed()

        ## STAGE 1 ##
        print(f"STAGE 1")
        for epoch in tqdm(range(epochs_stage1)):
            if torch.cuda.is_available():
                modelM.train()
                modelC.train()
            for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_id = y_id.cuda()
                # Compute prediction and loss
                prob = modelM(X.float())
                prob = modelC(prob)
                loss = loss_fn(prob,y_id.long())

                # Backpropagation
                optimizer_stage1.zero_grad()
                loss.backward()
                optimizer_stage1.step()

        ## STAGE 2 ##
        print(f"STAGE 2")
        for i in tqdm(range(Nep)):

            if i % Tep == 0:
                if torch.cuda.is_available():
                    modelE = NeuralNetworkE(n_subgroup).cuda()
                optimizer_stage2 = optim.Adam(modelE.parameters(), lr=1e-3)
    #             print(f"STAGE 2")
                for epoch in range(epochs_stage2):
                    for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                        if torch.cuda.is_available():
                            X = X.cuda()
                            y_subgroup = y_subgroup.cuda()
                        prob = modelM(X.float())
                        prob = modelE(prob)
                        loss = loss_fn(prob,y_subgroup.long())

                        # Backpropagation
                        optimizer_stage2.zero_grad()
                        loss.backward()
                        optimizer_stage2.step()

            ## STAGE 3 ##
            optimizer_stage3 = optim.Adam(list(modelM.parameters())+list(modelC.parameters()), lr=1e-3)
        #     print(f"STAGE 3")
            for epoch in range(epochs_stage3):
                for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y_id = y_id.cuda()
                        y_subgroup = y_subgroup.cuda()
                    f_out = modelM(X.float())
                    prob_class = modelC(f_out)
                    prob_subgroup = modelE(f_out)

                    loss_class = loss_fn(prob_class,y_id.long())

                    loss_deb = -torch.log(prob_subgroup)/prob_subgroup.shape[1]
                    loss_deb = loss_deb.sum(axis=1).mean()

                    loss = loss_class+10*loss_deb

                    # Backpropagation
                    optimizer_stage3.zero_grad()
                    loss.backward()
                    optimizer_stage3.step()

            ## STAGE 4 ##

            optimizer_stage2 = optim.Adam(modelE.parameters(), lr=1e-3)
        #     print(f"STAGE 4")
            for epoch in range(epochs_stage4):
                modelM.eval()
                modelE.eval()
                size = len(test_dataloader.dataset)
                test_loss, correct = 0, 0

                scores = torch.zeros(0, 2)
                ground_truth = torch.zeros(0)
                with torch.no_grad():
                    for X, y_id, y_subgroup in test_dataloader:
                        if torch.cuda.is_available():
                            X = X.cuda()
                            y_subgroup = y_subgroup.cuda()
                        prob = modelM(X.float())
                        prob = modelE(prob)
                        test_loss += loss_fn(prob,y_subgroup.long()).item()
                        correct += (prob.argmax(1) == y_subgroup).type(torch.float).sum().item()
                test_loss /= size
                correct /= size
        #         print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")

                if correct > 0.9:
                    break
                for batch, (X, y_id, y_subgroup) in enumerate(train_dataloader):
                    if torch.cuda.is_available():
                        X = X.cuda()
                        y_id = y_id.cuda()
                        y_subgroup = y_subgroup.cuda()
                    prob = modelM(X.float())
                    prob = modelE(prob)
                    loss = loss_fn(prob,y_subgroup.long())

                    # Backpropagation
                    optimizer_stage2.zero_grad()
                    loss.backward()
                    optimizer_stage2.step()

        fair_scores = {}
        ground_truth = {}
        for dataset in ['cal', 'test']:
            embeddings, ground_truth[dataset], subgroups_left, subgroups_right = self.collect_pair_embeddings(db_fold[dataset])
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            modelM.eval()
            modelM.cpu()
            with torch.no_grad():
                temp1 = modelM(embeddings['left'])
                temp2 = modelM(embeddings['right'])
            output = cos(temp1, temp2)
            fair_scores[dataset] = output.numpy()

        confidences = self.baseline(fair_scores, ground_truth, score_min=-1, score_max=1)

        return fair_scores, confidences, modelM, modelC, modelE

    def collect_embeddings_agenda(self, db_cal, embedding_data):
        """
        Placeholder method to be overwritten.
        """
        return None, None, None

    def collect_pair_embeddings(self, db_cal):
        """
        Placeholder method to be overwritten.
        """
        return None, None, None, None

    def baseline(self, fair_scores, ground_truth, score_min, score_max):
        """
        Placeholder method to be overwritten.
        """
        return None


class FtcApproach:
    dataset = None
    def ftc(self, db_fold):
        """
        Method that implements the FTC approach.
        """
        error_embeddings, ground_truth, subgroups_left, subgroups_right = self.collect_error_embeddings(db_fold['cal'])
        train_dataloader = DataLoader(
            FtcEmbeddingsDataset(error_embeddings, ground_truth, subgroups_left, subgroups_right),
            batch_size=200,
            shuffle=True,
            num_workers=0)
        # We note that this code could be improved by using a true validation set.
        evaluate_train_dataloader = DataLoader(
            FtcEmbeddingsDataset(error_embeddings, ground_truth, subgroups_left, subgroups_right),
            batch_size=200,
            shuffle=False,
            num_workers=0)

        error_embeddings, ground_truth, subgroups_left, subgroups_right = self.collect_error_embeddings(db_fold['test'])

        evaluate_test_dataloader = DataLoader(
            FtcEmbeddingsDataset(error_embeddings, ground_truth, subgroups_left, subgroups_right),
            batch_size=200,
            shuffle=False,
            num_workers=0)

        set_seed()

        # Initialize model
        model = NeuralNetwork()
        # Initialize the loss function
        loss_fn = nn.CrossEntropyLoss()
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        # Set the number of epochs
        epochs = 50
        # Set the seed
        set_seed()
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, self.dataset)
            _, _ = test_loop(evaluate_test_dataloader, model, loss_fn)
        print("Done!")
        scores_cal, ground_truth_cal = test_loop(evaluate_train_dataloader, model, loss_fn)
        scores_cal = scores_cal[:, 1].numpy().reshape(-1)
        assert sum(np.array(ground_truth_cal == 1) != np.array(db_fold['cal']['same'])) == 0

        scores_test, ground_truth_test = test_loop(evaluate_test_dataloader, model, loss_fn)
        scores_test = scores_test[:, 1].numpy().reshape(-1)
        assert sum(np.array(ground_truth_test == 1) != np.array(db_fold['test']['same'])) == 0

        fair_scores = {'cal': scores_cal, 'test': scores_test}
        ground_truth = {'cal': ground_truth_cal, 'test': ground_truth_test}

        confidences = self.baseline(fair_scores, ground_truth, score_min=-1, score_max=1)

        return fair_scores, confidences, model

    def baseline(self, fair_scores, ground_truth, score_min, score_max):
        """
        Placeholder method to be overwritten.
        """
        return None

    def collect_error_embeddings(self, db_cal):
        """
        Placeholder method to be overwritten.
        """
        return None, None, None, None


class ApproachManager(AgendaApproach, FtcApproach):
    """
    Class that contains all the functionality relating to the different approaches.

    It inherits from AgendaApproach and FtcApproach methods that are specific to the Agenda and FTC approach
    respectively.
    """
    nbins = None
    dataset = None
    approach = None
    calibration_method = None
    feature = None
    n_cluster = None
    fpr_thr = None
    subgroups = None

    def get_calibration_method(self, scores, ground_truth, score_min=-1, score_max=1):
        """
        This method returns the calibration. Three different calibration methods are supported:
        binning, isotonic_regression and beta.
        """
        # Extract if embedding in a dictionary
        if isinstance(scores, np.ndarray):
            score = scores
            gt = ground_truth
        else:
            score = scores['cal']
            gt = ground_truth['cal']
        if self.calibration_method == 'binning':
            calibration = BinningCalibration(
                score, gt, score_min=score_min, score_max=score_max, nbins=self.nbins
            )
        elif self.calibration_method == 'isotonic_regression':
            calibration = IsotonicCalibration(
                score, gt, score_min=score_min, score_max=score_max
            )
        elif self.calibration_method == 'beta':
            calibration = BetaCalibration(score, gt, score_min=score_min, score_max=score_max)
        else:
            raise ValueError('Calibration method %s not available' % self.calibration_method)
        return calibration

    def baseline(self, scores, ground_truth, score_min=-1, score_max=1):
        """
        This is the baseline approach that derives the confidences.
        """
        calibration = self.get_calibration_method(scores, ground_truth, score_min, score_max)
        return {'cal': calibration.predict(scores['cal']), 'test': calibration.predict(scores['test'])}

    def oracle(self, scores, ground_truth, subgroup_scores):
        """
        This is the oracle approach that derives the confidences.
        """
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

    def cluster_methods(self, fold, db_fold, score_normalization, fpr, embedding_data, seed=0):
        """
        This method contains the clustering and score calculations for the Faircal, FSN and Faircal-GMM.

        Faircal and FSN use k-means clustering, whilst Faircal-GMM used Gaussian mixtures of models.
        """
        # k-means algorithm
        saveto = (
            f"experiments/clustering_{self.approach}/{self.dataset}_{self.feature}_nclusters{self.n_cluster}_fold{fold}"
            ".npy"
        )
        if os.path.exists(saveto):
            os.remove(saveto)
        prepare_dir(saveto)
        np.save(saveto, {})
        embeddings = self.collect_embeddings(db_fold['cal'], embedding_data)

        if self.approach in ('faircal', 'fsn'):
            cluster_model = self.kmeans_clustering(embeddings)
        elif self.approach == 'faircal-gmm':
            cluster_model = self.gmm_clustering(embeddings)
        else:
            raise ValueError(f'Unrecognised approach: {self.approach}')

        cluster_model.fit(embeddings.astype('float32'))
        prepare_dir(saveto)
        np.save(saveto, cluster_model)

        r = self.collect_miscellania(self.n_cluster, cluster_model, db_fold, embedding_data)

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

    def kmeans_clustering(self, embeddings):
        set_seed()
        gpu_bool = torch.cuda.is_available()
        if gpu_bool:
            cluster_method = KMeans(num_clusters=self.n_cluster, trainer_params=dict(accelerator='gpu', devices=1))
        else:
            cluster_method = KMeans(num_clusters=self.n_cluster)
        cluster_method.fit(embeddings.astype('float32'))

        return cluster_method

    def gmm_clustering(self, embeddings, seed=0):
        set_seed(seed)
        gpu_bool = torch.cuda.is_available()
        try:
            if gpu_bool:
                cluster_method = GaussianMixture(
                    num_components=self.n_cluster, trainer_params=dict(accelerator='gpu', devices=1),
                    covariance_type='full'
                )
            else:
                cluster_method = GaussianMixture(num_components=self.n_cluster, covariance_type='full')
            cluster_method.fit(embeddings.astype('float32'))
        except Exception as e:
            print(f'An exception occured: {e}')
            print('PyCave GMM failed. Defaulting back to sklearn GMM. This will take longer...')
            cluster_method = SkGaussianMixture(self.n_cluster, random_state=seed)

            cluster_method.fit(embeddings.astype('float32'))

        return cluster_method

    def get_metrics(
            self, embedding_data, db_fold, fold, scores, ground_truth, subgroup_scores
    ):
        """
        This method is to obtain the metrics for the different approaches:
        ['baseline', 'faircal', 'fsn', 'agenda', 'faircal-gmm', 'oracle']
        """
        fair_scores = None
        if self.approach == 'baseline':
            confidences = self.baseline(scores, ground_truth)
        elif self.approach in ('faircal', 'faircal-gmm'):
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
        """
        This method is used to find the thresholds for the binary classification.
        """
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


class AgendaEmbeddingsDataset(Dataset):
    """
    Embeddings dataset for the Agenda approach.
    """

    def __init__(self, embeddings, id_embeddings, subgroup_embeddings):
        """
        Arguments
        """
        self.embeddings = embeddings
        self.id_embeddings = id_embeddings
        self.subgroup_embeddings = subgroup_embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx, :], self.id_embeddings[idx], self.subgroup_embeddings[idx]


class NeuralNetworkM(nn.Module):
    """
    Fully connected NN for the Agenda approach.
    """
    def __init__(self):
        super(NeuralNetworkM, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)

class NeuralNetworkC(nn.Module):
    """
    Fully connected NN for the Agenda approach.
    """
    def __init__(self,nClasses):
        super(NeuralNetworkC, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, nClasses)
        )

    def forward(self, x):
        return self.model(x)


class NeuralNetworkE(nn.Module):
    """
    Fully connected NN for the Agenda approach.
    """
    def __init__(self,nClasses):
        super(NeuralNetworkE, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128, nClasses),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


class NeuralNetwork(nn.Module):
    """
    Fully connected NN for the Agenda approach.
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128*4, 256*4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256*4, 512*4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512*4, 512*4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512*4, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        prob = self.softmax(logits)
        return logits, prob


class FtcEmbeddingsDataset(Dataset):
    """
    Embeddings dataset for the FTC approach.
    """

    def __init__(self, error_embeddings, ground_truth, subgroups_left, subgroups_right):
        """
        Arguments
        """
        self.subgroups_left = subgroups_left
        self.subgroups_right = subgroups_right
        self.error_embeddings = error_embeddings
        self.labels = torch.zeros(len(error_embeddings)).type(torch.LongTensor)
        self.labels[ground_truth] = 1

    def __len__(self):
        return len(self.error_embeddings)

    def __getitem__(self, idx):
        return self.error_embeddings[idx, :], self.subgroups_left[idx], self.subgroups_right[idx], self.labels[idx]


def fair_individual_loss(g1, g2, y, yhat, dataset_name):
    """
    Fair individual loss function used for the FTC approach.
    """
    if dataset_name == 'rfw':
        subgroups = ['Asian', 'African', 'Caucasian', 'Indian']
    elif 'bfw' in dataset_name:
        subgroups = ['asian_females', 'asian_males', 'black_females', 'black_males', 'indian_females', 'indian_males',
                     'white_females', 'white_males']
    else:
        subgroups = None
    loss = 0
    for i in subgroups:
        for j in subgroups:
            select_i = np.logical_and(np.array(g1) == i, np.array(g2) == i)
            select_j = np.logical_and(np.array(g1) == j, np.array(g2) == j)
            if (sum(select_i) > 0) and (sum(select_j) > 0):
                select = y[select_i].reshape(-1, 1) == y[select_j]
                aux = torch.cdist(yhat[select_i, :], yhat[select_j, :])[select].pow(2).sum()
                loss += aux/(sum(select_i)*sum(select_j))
    return loss


def train_loop(dataloader, model, loss_fn, optimizer, dataset_name):
    """
    Training loop function used for the FTC approach.
    """
    set_seed()
    if dataset_name == 'rfw':
        batch_check = 50
    elif 'bfw' in dataset_name:
        batch_check = 500
    model.cuda()
    size = len(dataloader.dataset)
    for batch, (X, g1, g2, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        # Compute prediction and loss
        pred, prob = model(X)
        loss = 0.5*loss_fn(pred, y)+0.5*fair_individual_loss(g1, g2, y, pred, dataset_name)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % batch_check == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    model.cpu()


def test_loop(dataloader, model, loss_fn):
    """
    Testing loop function used for the FTC approach.
    """
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    scores = torch.zeros(0, 2)
    ground_truth = torch.zeros(0)
    with torch.no_grad():
        for X, g1, g2, y in dataloader:
            pred, prob = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            scores = torch.cat((scores, prob))
            ground_truth = torch.cat([ground_truth, y], 0)
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    fpr, tpr, thr = roc_curve(ground_truth, scores[:, 1].numpy())
    print('FNR @ 0.1 FPR %1.2f'% (1-tpr[np.argmin(np.abs(fpr-1e-3))]))
    return scores, ground_truth
