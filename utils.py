import numpy as np
from sklearn.metrics import roc_curve
import os
import torch
import sys
import pickle


def save_outputs(data_to_save, output_folder, model, dataset, limit_images=None):
    save_str = os.path.join(output_folder, f"{model}_{dataset}")
    if limit_images is not None:
        save_str = save_str + f"_limited_{limit_images}"
    prepare_dir(save_str)
    for k, df in data_to_save.items():
        pickle.dump(df, open(f"{save_str}_{k}.pk", "wb"))
    return save_str

def determine_device(cpu_bool):
    if cpu_bool:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using device {device}')
    return device


def prepare_dir(file_path):
    dir_path = '/'.join(file_path.split('/')[:-1])
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass


def batch(iterable, n=1):
    """
	https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
	"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def determine_edges(scores, nbins, indicesQ=False, score_min=-1, score_max=1):
    if nbins > len(scores):
        print('More bins than values. Attempting to return bins with one value each.')
        return determine_edges(scores, len(scores), indicesQ=indicesQ, score_min=score_min, score_max=score_max)
    scores_sorted = scores.copy()
    scores_index = scores.argsort()
    scores_sorted = scores_sorted[scores_index]
    aux = np.linspace(0, len(scores_sorted) - 1, nbins + 1).astype(int) + 1
    bin_indices = np.zeros(len(scores_sorted)).astype(int)
    bin_indices[:aux[1]] = 0
    for i in range(1, len(aux) - 1):
        bin_indices[aux[i]:aux[i + 1]] = i
    bin_edges = np.zeros(nbins + 1)
    for i in range(0, nbins - 1):
        bin_edges[i + 1] = np.mean(np.concatenate((
            scores_sorted[bin_indices == i][scores_sorted[bin_indices == i] == max(scores_sorted[bin_indices == i])],
            scores_sorted[bin_indices == (i + 1)][
                scores_sorted[bin_indices == (i + 1)] == min(scores_sorted[bin_indices == (i + 1)])])))
    bin_edges[0] = score_min
    bin_edges[-1] = score_max
    bin_indices = bin_indices[np.argsort(scores_index)]
    if len(np.unique(bin_edges)) < len(bin_edges):
        return determine_edges(scores, nbins - 1, indicesQ=indicesQ, score_min=score_min, score_max=score_max)
    else:
        if indicesQ:
            return bin_edges, bin_indices
        else:
            return bin_edges


def bin_confidences_and_accuracies(confidences, ground_truth, bin_edges, indices):
    i = np.arange(0, bin_edges.size-1)
    aux = indices == i.reshape((-1, 1))
    counts = aux.sum(axis=1)
    weights = counts / np.sum(counts)
    correct = np.logical_and(aux, ground_truth).sum(axis=1)
    a = np.repeat(confidences.reshape(1, -1), bin_edges.size-1, axis=0)
    a[np.logical_not(aux)] = 0
    bin_accuracy = correct / counts
    bin_confidence = a.sum(axis=1) / counts
    return weights, bin_accuracy, bin_confidence


def get_ks(confidences, ground_truth):
    n = len(ground_truth)
    order_sort = np.argsort(confidences)
    ks = np.max(np.abs(np.cumsum(confidences[order_sort])/n-np.cumsum(ground_truth[order_sort])/n))
    return ks


def get_brier(confidences, ground_truth):
    # Compute Brier Score
    brier = np.zeros(confidences.shape)
    brier[ground_truth] = (1-confidences[ground_truth])**2
    brier[np.logical_not(ground_truth)] = (confidences[np.logical_not(ground_truth)])**2
    brier = np.mean(brier)
    return brier


def get_ece(confidences, ground_truth, nbins):
    # Repeated code from determine edges. Here it is okay if the bin edges are not uniquely defined
    confidences_sorted = confidences.copy()
    confidences_index = confidences.argsort()
    confidences_sorted = confidences_sorted[confidences_index]
    aux = np.linspace(0, len(confidences_sorted) - 1, nbins + 1).astype(int) + 1
    bin_indices = np.zeros(len(confidences_sorted)).astype(int)
    bin_indices[:aux[1]] = 0
    for i in range(1, len(aux) - 1):
        bin_indices[aux[i]:aux[i + 1]] = i
    bin_edges = np.zeros(nbins + 1)
    for i in range(0, nbins - 1):
        bin_edges[i + 1] = np.mean(np.concatenate((
            confidences_sorted[bin_indices == i][confidences_sorted[bin_indices == i] == max(confidences_sorted[bin_indices == i])],
            confidences_sorted[bin_indices == (i + 1)][
                confidences_sorted[bin_indices == (i + 1)] == min(confidences_sorted[bin_indices == (i + 1)])])))
    bin_edges[0] = 0
    bin_edges[-1] = 1
    bin_indices = bin_indices[np.argsort(confidences_index)]

    weights, bin_accuracy, bin_confidence = bin_confidences_and_accuracies(confidences, ground_truth, bin_edges,
                                                                           bin_indices)
    ece = np.dot(weights, np.abs(bin_confidence - bin_accuracy))
    return ece


def compute_scores(confidences, ground_truth, nbins):

    # Compute ECE
    ece = get_ece(confidences, ground_truth, nbins)

    # Compute Brier
    brier = get_brier(confidences, ground_truth)

    # Compute KS
    ks = get_ks(confidences, ground_truth)

    return ece, ks, brier


def get_binary_clf_stats(ground_truth, scores):
    thr = np.unique(scores.copy())
    # false positives
    fp = np.sum(scores[np.logical_not(ground_truth)] >= thr.reshape(-1, 1), axis=1)
    # true negative
    tn = np.sum(scores[np.logical_not(ground_truth)] < thr.reshape(-1, 1), axis=1)
    # false negative
    fn = np.sum(scores[ground_truth] < thr.reshape(-1, 1), axis=1)
    # true positive
    tp = np.sum(scores[ground_truth] >= thr.reshape(-1, 1), axis=1)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    return fpr, tpr, thr


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class FileManager:
    @staticmethod
    def get_save_file_path(dataset, feature, approach, calibration_method, nbins, n_cluster, fpr_thr):
        experiments_folder = 'experiments/'
        folder_name = '/'.join([dataset, feature, approach, calibration_method])
        if 'faircal' in approach:
            file_name = '_'.join(['nbins', str(nbins), 'nclusters', str(n_cluster)])
        elif 'fsn' in approach:
            file_name = '_'.join(['nbins', str(nbins), 'nclusters', str(n_cluster), 'fpr', format(fpr_thr, '.0e')])
        else:
            file_name = '_'.join(['nbins', str(nbins)])
        return f"{experiments_folder}{folder_name}/{file_name}.npy"

    @staticmethod
    def prepare_output_dir(saveto):
        if os.path.exists(saveto):
            os.remove(saveto)
        prepare_dir(saveto)
        np.save(saveto, {})


class ExecuteSilently(object):
    """
    https://codereview.stackexchange.com/questions/25417/is-there-a-better-way-to-make-a-function-silent-on-need
    """
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()
