import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from csv_creator import CsvCreator
from generate_embeddings import save_outputs


def cosine_calc(model_name, dataset):
    if dataset == 'rfw':
        return rfw_cosine_calc(model_name)
    elif dataset == 'bfw':
        return bfw_cosine_calc(model_name)
    else:
        raise ValueError(f'Unrecognised dataset: {dataset}')


def rfw_cosine_calc(model_name):
    # Load data rfw ddata
    rfw_df = CsvCreator.get_rfw_df()

    # Create a column id to map the embeddings
    rfw_df['image_id_1_clean'] = rfw_df['id1'].map(str) + '_' + rfw_df['num1'].map(str)
    rfw_df['image_id_2_clean'] = rfw_df['id2'].map(str) + '_' + rfw_df['num2'].map(str)

    # Load the embedding data
    emb_df = pickle.load(open(f'./embeddings/{model_name}_rfw_embeddings.pk', 'rb'))
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


def bfw_cosine_calc(model_name):
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


def derive_cosine_sim_for_all_sets():
    very_start = time.time()
    task_list = [
        ('facenet-webface', 'bfw'),
        ('arcface', 'bfw'),
        ('facenet-webface', 'rfw'),
        ('facenet', 'rfw')
    ]
    for model, dataset in task_list:
        start = time.time()
        cos_df = cosine_calc(model, dataset)
        save_outputs({'cosin_sim': cos_df}, 'similarities', model, dataset)
        print(f'derive_cosin_sim_for_all_sets {model} {dataset} took: {round(time.time() - start)} seconds.')
    print(f'derive_cosin_sim_for_all_sets total took: {round(time.time() - very_start)} seconds.')

if __name__ == '__main__':
    derive_cosine_sim_for_all_sets()






