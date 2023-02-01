import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class AgendaApproach:
    """
    The AgendaApproach class contains all of the methods that are specific to the Agenda approach.
    """

    dataset = None
    def agenda(self, db_fold, embedding_data):

        embeddings, subgroup_embeddings, id_embeddings = self.collect_embeddings_agenda(db_fold['cal'], embedding_data)

        embeddings_train, embeddings_test, id_train, id_test, subgroup_train, subgroup_test \
            = train_test_split(embeddings, id_embeddings, subgroup_embeddings, test_size=0.2)

        id_train = pd.Series(id_train, dtype="category").cat.codes.values
        id_test = pd.Series(id_test, dtype="category").cat.codes.values

        train_dataloader = DataLoader(
                EmbeddingsDataset(embeddings_train, id_train, subgroup_train),
                batch_size=400,
                shuffle=True,
                num_workers=0)

        test_dataloader = DataLoader(
                EmbeddingsDataset(embeddings_test, id_test, subgroup_test),
                batch_size=400,
                shuffle=True,
                num_workers=0)

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

        optimizer_stage1 = optim.Adam(list(modelM.parameters())+list(modelC.parameters()), lr=1e-3)
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



class EmbeddingsDataset(Dataset):
    """Embeddings dataset."""

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
    def __init__(self):
        super(NeuralNetworkM, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)

class NeuralNetworkC(nn.Module):
    def __init__(self,nClasses):
        super(NeuralNetworkC, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, nClasses)
        )

    def forward(self, x):
        return self.model(x)

class NeuralNetworkE(nn.Module):
    def __init__(self,nClasses):
        super(NeuralNetworkE, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128,nClasses),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
