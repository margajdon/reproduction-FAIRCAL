import pandas as pd
from numpy.linalg import norm
import numpy as np
import pickle
import time

def main():
    # Load data
    embeddings = pickle.load(open('./embeddings/facenet-webface_bfw_limited_2000_embeddings.pk', 'rb'))[['img_path', 'embedding']]
    given = pd.read_csv('./data/bfw/bfw.csv')[['p1', 'p2', 'facenet']]

    # Make sure both dfs have same formatted paths
    embeddings['img_path'] = embeddings['img_path'].apply(lambda x: x.replace('data\\bfw\\faces-cropped\\', ''))
    embeddings['img_path'] = embeddings['img_path'].apply(lambda x: x.replace('\\', '/'))
    
    # Get only the images for which we have embeddings
    given = given[(given['p1'].isin(embeddings['img_path'])) & (given['p2'].isin(embeddings['img_path']))]
    print(f'Considering {len(given)} pairs')

    given['cos_sim'] = given.apply(lambda x: getEmbedding(x, embeddings), axis=1)
    difference = (given['cos_sim'] - given['facenet']).to_numpy()
    print(f'MAE: {round(np.abs(difference).mean(), 3)}')
    print(f'standard dev: {round(difference.std(), 3)}')

def getEmbedding(row, embeddings):
    embedding_p1 = embeddings[embeddings['img_path'] == row['p1']]['embedding'].to_numpy()[0]
    embedding_p2 = embeddings[embeddings['img_path'] == row['p2']]['embedding'].to_numpy()[0]
    return cos_sim(embedding_p1, embedding_p2)

def cos_sim(a, b):
    return np.dot(a,b) / (norm(a)*norm(b)) 

if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Took {round(time.time() - start)} seconds')