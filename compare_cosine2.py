import os

import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from generate_embeddings import save_outputs
from rfw_csv_creation import get_rfw_df
from utils import prepare_dir


import os

import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from generate_embeddings import save_outputs
from utils import prepare_dir


def rfw_comparison(model_name, dataset='rfw', show_plot=False):
    # Load data
    rfw_df = get_rfw_df()
    rfw_df['image_id_1_clean'] = rfw_df['id1'].map(str) + '_' + rfw_df['num1'].map(str)
    rfw_df['image_id_2_clean'] = rfw_df['id2'].map(str) + '_' + rfw_df['num2'].map(str)

    emb_df = pickle.load(open(f'./embeddings/{model_name}_{dataset}_embeddings.pk', 'rb'))
    emb_df['image_id_clean'] = emb_df['image_id'].map(lambda x: x.replace('000', ''))

    cond = (
        rfw_df['image_id_1_clean'].isin(emb_df['image_id_clean'].tolist())
        &
        rfw_df['image_id_2_clean'].isin(emb_df['image_id_clean'].tolist())
    )
    base_df2 = rfw_df[cond].reset_index(drop=True)
    embedding_map = dict(zip(emb_df['image_id_clean'], emb_df['embedding']))
    base_df2['embedding1'] = base_df2['image_id_1_clean'].map(embedding_map)
    base_df2['embedding2'] = base_df2['image_id_2_clean'].map(embedding_map)
    base_df2['embedding_pair'] = list(zip(base_df2['embedding1'], base_df2['embedding2']))
    base_df2['cos_sim'] = base_df2['embedding_pair'].map(lambda x: cos_sim(x[0], x[1]))

    get_cross_entropy_loss(base_df2['same'], base_df2['cos_sim'])


    cols_to_return = ['id1', 'num1', 'id2', 'num2', 'ethnicity', 'pair', 'same', 'fold',
       'facenet', 'facenet-webface', 'arcface', 'image_id_1_clean',
       'image_id_2_clean', 'embedding_pair',
       'cos_sim'
    ]

    return base_df2[cols_to_return]


def bfw_comparison(model_name, show_plot=False):
    # Load data
    base_df = pd.read_csv('./data/bfw/bfw.csv')
    base_df['image_id_1_clean'] = base_df['p1'].map(lambda x: '_'.join(x[:-4].split('/')[-2:]))
    base_df['image_id_2_clean'] = base_df['p2'].map(lambda x: '_'.join(x[:-4].split('/')[-2:]))
    base_df['id_combined'] = base_df['image_id_1_clean'] + '_&_' + base_df['image_id_2_clean']
    base_df['label'] = base_df['same']

    emb_df = pickle.load(open(f'./embeddings/{model_name}_bfw_embeddings.pk', 'rb'))
    emb_df['image_id_clean'] = emb_df['person'].map(str) + '_' + emb_df['image_id'].map(str)

    cond = (
        base_df['image_id_1_clean'].isin(emb_df['image_id_clean'].tolist())
        &
        base_df['image_id_2_clean'].isin(emb_df['image_id_clean'].tolist())
    )
    base_df2 = base_df[cond].reset_index(drop=True)
    embedding_map = dict(zip(emb_df['image_id_clean'], emb_df['embedding']))
    base_df2['embedding1'] = base_df2['image_id_1_clean'].map(embedding_map)
    base_df2['embedding2'] = base_df2['image_id_2_clean'].map(embedding_map)
    base_df2['embedding_pair'] = list(zip(base_df2['embedding1'], base_df2['embedding2']))
    base_df2['cos_sim'] = base_df2['embedding_pair'].map(lambda x: cos_sim(x[0], x[1]))

    # get_cross_entropy_loss(base_df2['same'], base_df2['cos_sim'])
    #
    # get_cross_entropy_loss(base_df2['same'], base_df2['facenet'])
    # get_cross_entropy_loss(base_df2['same'], base_df2['senet50'])

    # Calculate difference and report metrics
    difference = (base_df2['cos_sim'] - base_df2['facenet']).to_numpy()
    mean = round(difference.mean(), 3)
    std = round(difference.std(), 3)
    corr = round(base_df2[["cos_sim", "facenet"]].corr()["cos_sim"]["facenet"], 3)

    print("Comparing cosine similarity of photo-pair embeddings")
    print(f'Mean: {mean}')
    print(f'Std: {std}')
    print(f'Linear corr: {corr}')

    if show_plot:

        difference_same = (base_df2[base_df2["label"] == 1]['cos_sim'] - base_df2[base_df2["label"] == 1]['facenet']).to_numpy()
        difference_diff = (base_df2[base_df2["label"] == 0]['cos_sim'] - base_df2[base_df2["label"] == 0]['facenet']).to_numpy()

        plt.hist(difference_same, bins=40, alpha=0.7)
        plt.hist(difference_diff, bins=40, alpha=0.7)
        plt.plot([], [], ' ', label=f'n={len(difference)}\nMean={mean}\nStd={std}\nCorr={corr}')
        plt.xlabel("predicted - reference")
        plt.ylabel("frequency")
        plt.legend(handletextpad=.0, handlelength=0)
        plt.title("Comparison of cosine similarity of photo-pair embeddings")
        plt.show()

    cols_to_return = [
        'fold', 'p1', 'p2', 'same', 'id1', 'id2', 'att1', 'att2', 'facenet', 'resnet50', 'senet50', 'a1', 'a2', 'g1',
        'g2', 'e1', 'e2', 'image_id_1_clean', 'image_id_2_clean', 'id_combined', 'label', "cos_sim",
    ]

    return base_df2[cols_to_return]


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_cross_entropy_loss(y_label, y_pred):
    y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(
        (1 - y_label) * np.log(1 - y_pred_clipped + 1e-9)
        +
        y_label * np.log(y_pred_clipped + 1e-9),
        axis=0
    )




if __name__ == '__main__':
    start = time.time()
    model = 'facenet-webface'
    cos_df = bfw_comparison(model)
    print(f'Took {round(time.time() - start)} seconds')

    # save_outputs({'cosin_sim': cos_df}, 'similarities', model, 'bfw')

    model = 'arcface'

    start = time.time()
    cos_df = bfw_comparison(model)
    print(f'Took {round(time.time() - start)} seconds')
    save_outputs({'cosin_sim': cos_df}, 'similarities', model, 'bfw')


    model = 'facenet'
    cos_df = rfw_comparison(model, 'rfw')
    save_outputs({'cosin_sim': cos_df}, 'similarities', model, 'rfw')

    model = 'facenet-webface'
    cos_df = rfw_comparison(model, 'rfw')
    save_outputs({'cosin_sim': cos_df}, 'similarities', model, 'rfw')




