from rfw_csv_creation import get_rfw_df
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from generate_embeddings import save_outputs


def rfw_comparison(model_name, dataset='rfw', show_plot=False):
    # Load data rfw ddata
    rfw_df = get_rfw_df()

    # Create a column id to map the embeddings
    rfw_df['image_id_1_clean'] = rfw_df['id1'].map(str) + '_' + rfw_df['num1'].map(str)
    rfw_df['image_id_2_clean'] = rfw_df['id2'].map(str) + '_' + rfw_df['num2'].map(str)

    # Load the embedding data
    emb_df = pickle.load(open(f'./embeddings/{model_name}_{dataset}_embeddings.pk', 'rb'))
    emb_df['image_id_clean'] = emb_df['image_id'].map(lambda x: x.replace('000', ''))

    # Filter out rows that don't have an embedding
    images_w_avail_embeddings = emb_df['image_id_clean'].tolist()
    rfw_clean = rfw_df[
        rfw_df['image_id_1_clean'].isin(images_w_avail_embeddings) &
        rfw_df['image_id_2_clean'].isin(images_w_avail_embeddings)
    ].reset_index(drop=True)

    # Map the embedding to the rfw file
    embedding_map = dict(zip(emb_df['image_id_clean'], emb_df['embedding']))
    rfw_clean['embedding1'] = rfw_clean['image_id_1_clean'].map(embedding_map)
    rfw_clean['embedding2'] = rfw_clean['image_id_2_clean'].map(embedding_map)
    
    # Combine the embedding pair in one column for efficient compute
    rfw_clean['embedding_pair'] = list(zip(rfw_clean['embedding1'], rfw_clean['embedding2']))

    # Calculate cosine similarity
    rfw_clean['cos_sim'] = rfw_clean['embedding_pair'].map(lambda x: cos_sim(x[0], x[1]))

    # Create a list with the columns we want to return
    cols_to_return = [c for c in rfw_clean.columns if c not in ('embedding1', 'embedding2')]
    
    # Return the df with the selected columns
    return rfw_clean[cols_to_return]


def bfw_comparison(model_name, show_plot=False):
    # Load data bfw ddata
    bfw_df = pd.read_csv('./data/bfw/bfw.csv')

    # Create a column id to map the embeddings
    bfw_df['image_id_1_clean'] = bfw_df['p1'].map(lambda x: '_'.join(x[:-4].split('/')[-2:]))
    bfw_df['image_id_2_clean'] = bfw_df['p2'].map(lambda x: '_'.join(x[:-4].split('/')[-2:]))
    
    # Rename a column
    bfw_df = bfw_df.rename(columns={'same': 'label'})
    
    # Load the embeddings
    emb_df = pickle.load(open(f'./embeddings/{model_name}_bfw_embeddings.pk', 'rb'))
    
    # Create column id to map the embeddings
    emb_df['image_id_clean'] = emb_df['person'].map(str) + '_' + emb_df['image_id'].map(str)

    # Filter the unneeded rows
    bfw_clean = bfw_df[
        bfw_df['image_id_1_clean'].isin(emb_df['image_id_clean'].tolist()) &
        bfw_df['image_id_2_clean'].isin(emb_df['image_id_clean'].tolist())
    ].reset_index(drop=True)

    # Create a dictionary to map the embeddings
    embedding_map = dict(zip(emb_df['image_id_clean'], emb_df['embedding']))

    # Mapp the mb
    bfw_clean['embedding1'] = bfw_clean['image_id_1_clean'].map(embedding_map)
    bfw_clean['embedding2'] = bfw_clean['image_id_2_clean'].map(embedding_map)

    # Combine the embedding pair in one column for efficient compute
    bfw_clean['embedding_pair'] = list(zip(bfw_clean['embedding1'], bfw_clean['embedding2']))

    # Calculate the cosine similarities of the pair
    bfw_clean['cos_sim'] = bfw_clean['embedding_pair'].map(lambda x: cos_sim(x[0], x[1]))


    if show_plot:

        # Calculate difference and report metrics
        difference = (bfw_clean['cos_sim'] - bfw_clean['facenet']).to_numpy()
        mean = round(difference.mean(), 3)
        std = round(difference.std(), 3)
        corr = round(bfw_clean[["cos_sim", "facenet"]].corr()["cos_sim"]["facenet"], 3)

        print("Comparing cosine similarity of photo-pair embeddings")
        print(f'Mean: {mean}')
        print(f'Std: {std}')
        print(f'Linear corr: {corr}')

        difference_same = (bfw_clean[bfw_clean["label"] == 1]['cos_sim'] - bfw_clean[bfw_clean["label"] == 1]['facenet']).to_numpy()
        difference_diff = (bfw_clean[bfw_clean["label"] == 0]['cos_sim'] - bfw_clean[bfw_clean["label"] == 0]['facenet']).to_numpy()

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

    # Create a list with the columns we want to return
    cols_to_return = [c for c in bfw_clean.columns if c not in ('embedding1', 'embedding2')]

    return bfw_clean[cols_to_return]


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

    # BFW - Webface
    model = 'facenet-webface'
    cos_df = bfw_comparison(model)
    save_outputs({'cosin_sim': cos_df}, 'similarities', model, 'bfw')

    # BFW - Arcface
    model = 'arcface'
    cos_df = bfw_comparison(model)
    save_outputs({'cosin_sim': cos_df}, 'similarities', model, 'bfw')

    # RFW - Facenet
    model = 'facenet'
    cos_df = rfw_comparison(model, 'rfw')
    save_outputs({'cosin_sim': cos_df}, 'similarities', model, 'rfw')

    # RFW - Webface
    model = 'facenet-webface'
    cos_df = rfw_comparison(model, 'rfw')
    save_outputs({'cosin_sim': cos_df}, 'similarities', model, 'rfw')




