import pandas as pd
from numpy.linalg import norm
import numpy as np
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    # Load data
    embeddings = pickle.load(open('./embeddings/facenet_bfw_embeddings.pk', 'rb'))[['img_path', 'embedding']]
    given = pd.read_csv('./data/bfw/bfw.csv')[['p1', 'p2', 'facenet']]
    print('Total reference cosine pairs:', len(given))

    # Make sure both dfs have same formatted paths
    embeddings['img_path'] = embeddings['img_path'].apply(lambda x: x.replace('data\\bfw\\faces-cropped\\', ''))
    embeddings['img_path'] = embeddings['img_path'].apply(lambda x: x.replace('\\', '/'))
    
    # Get only the images for which we have embeddings
    given = given[(given['p1'].isin(embeddings['img_path'])) & (given['p2'].isin(embeddings['img_path']))]
    given = given.sample(10000)
    print(f'Considering {len(given)} pairs')

    # Get cosine similarity
    sims = []
    for index, row in tqdm(given.iterrows(), total=len(given)):
        sims.append(getCosSim(row, embeddings))
    given['cos_sim'] = sims

    # Calculate difference and report metrics
    difference = (given['cos_sim'] - given['facenet']).to_numpy()
    print("Difference (predicted-reference):")
    print(f'Mean: {round(difference.mean(), 3)}')
    print(f'Std: {round(difference.std(), 3)}')
    print(f'Linear corr: {round(given[["cos_sim","facenet"]].corr()["cos_sim"]["facenet"], 3)}')
    plt.hist(difference, bins=40)
    plt.title("Difference (predicted-reference) cosine similarity")
    plt.show()

def getCosSim(row, embeddings):
    embedding_p1 = embeddings[embeddings['img_path'] == row['p1']]['embedding'].to_numpy()[0]
    embedding_p2 = embeddings[embeddings['img_path'] == row['p2']]['embedding'].to_numpy()[0]
    return cos_sim(embedding_p1, embedding_p2)

def cos_sim(a, b):
    return np.dot(a,b) / (norm(a)*norm(b)) 

if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Took {round(time.time() - start)} seconds')