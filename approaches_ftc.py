import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve

class FtcApproach:
    dataset = None
    def ftc(self, db_fold):
        r = self.collect_error_embeddings(db_fold['cal'])

        error_embeddings = r[0]
        ground_truth = r[1]
        subgroups_left = r[2]
        subgroups_right = r[3]
        train_dataloader = DataLoader(
            EmbeddingsDataset(error_embeddings, ground_truth, subgroups_left, subgroups_right),
            batch_size=200,
            shuffle=True,
            num_workers=0)
        evaluate_train_dataloader = DataLoader(
            EmbeddingsDataset(error_embeddings, ground_truth, subgroups_left, subgroups_right),
            batch_size=200,
            shuffle=False,
            num_workers=0)

        r = self.collect_error_embeddings(db_fold['test'])

        error_embeddings = r[0]
        ground_truth = r[1]
        subgroups_left = r[2]
        subgroups_right = r[3]

        evaluate_test_dataloader = DataLoader(
            EmbeddingsDataset(error_embeddings, ground_truth, subgroups_left, subgroups_right),
            batch_size=200,
            shuffle=False,
            num_workers=0)

        # Initialize model
        model = NeuralNetwork()
        # Initialize the loss function
        loss_fn = nn.CrossEntropyLoss()
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        epochs = 50
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


class NeuralNetwork(nn.Module):
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


class EmbeddingsDataset(Dataset):
    """Embeddings dataset."""

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
