import pandas as pd
from numpy.linalg import norm
import numpy as np
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def sim_histograms(filename):
    # Calculate difference and report metrics
    sims = pd.read_csv(filename)
    print(sims.columns)
    sims['label'] = sims['label'].astype(int)
    print('Total reference cosine pairs:', len(sims))

    mean = round(sims["cos_sim"].mean(), 3)
    std = round(sims["cos_sim"].std(), 3)
    
    print("Comparing cosine similarity of photo-pair embeddings")
    print(f'Mean: {mean}')
    print(f'Std: {std}')

    fig, axs = plt.subplots(1,2)
    


    sims_same = sims[sims["id1"]==sims["id2"]]["cos_sim"].to_numpy()
    sims_diff = sims[sims["id1"]!=sims["id2"]]["cos_sim"].to_numpy()
    axs[0].hist(sims_same, bins=100, alpha=0.7)
    axs[0].hist(sims_diff, bins=100, alpha=0.7)

    axs[0].plot([],[],' ', label=f'n={len(sims)}\nMean={mean}\nStd={std}')
    axs[0].set_xlabel("Cosine similarity")
    axs[0].set_ylabel("frequency")
    axs[0].legend(handletextpad=.0, handlelength=0)
    axs[0].set_title('sims["id1"]==sims["id2"]')



    sims_same = sims[sims["label"]==1]["cos_sim"].to_numpy()
    sims_diff = sims[sims["label"]==0]["cos_sim"].to_numpy()
    axs[1].hist(sims_same, bins=100, alpha=0.7)
    axs[1].hist(sims_diff, bins=100, alpha=0.7)

    axs[1].plot([],[],' ', label=f'n={len(sims)}\nMean={mean}\nStd={std}')
    axs[1].set_xlabel("Cosine similarity")
    axs[1].set_ylabel("frequency")
    axs[1].legend(handletextpad=.0, handlelength=0)
    axs[1].set_title('sims["label"]==1')


    plt.show()

def sim_histograms_with_existing_file(filename, model):
    # Calculate difference and report metrics
    sims = pd.read_csv(filename)
    print(sims.columns)
    sims['same'] = sims['same'].astype(int)
    print('Total reference cosine pairs:', len(sims))

    mean = round(sims[model].mean(), 3)
    std = round(sims[model].std(), 3)
    
    print("Comparing cosine similarity of photo-pair embeddings")
    print(f'Mean: {mean}')
    print(f'Std: {std}')

    fig, axs = plt.subplots(1,2)
    
    sims_diff = sims[sims["id1"]!=sims["id2"]][model].to_numpy()
    sims_same = sims[sims["id1"]==sims["id2"]][model].to_numpy()
    axs[0].hist(sims_same, bins=100, alpha=0.7)
    axs[0].hist(sims_diff, bins=100, alpha=0.7)

    axs[0].plot([],[],' ', label=f'n={len(sims)}\nMean={mean}\nStd={std}')
    axs[0].set_xlabel("Cosine similarity")
    axs[0].set_ylabel("frequency")
    axs[0].legend(handletextpad=.0, handlelength=0)
    # axs[0].set_title(f"Distribution of cos_sim ({filename.split('/')[-1]} - {model})")

    sims_same = sims[sims["same"]==1][model].to_numpy()
    sims_diff = sims[sims["same"]==0][model].to_numpy()
    axs[1].hist(sims_same, bins=100, alpha=0.7)
    axs[1].hist(sims_diff, bins=100, alpha=0.7)

    axs[1].plot([],[],' ', label=f'n={len(sims)}\nMean={mean}\nStd={std}')
    axs[1].set_xlabel("Cosine similarity")
    axs[1].set_ylabel("frequency")
    axs[1].legend(handletextpad=.0, handlelength=0)
    # axs[1].set_title(f"Distribution of cos_sim ({filename.split('/')[-1]} - {model})")
    plt.show()

def main():
    # Load data
    embeddings = pickle.load(open('./embeddings/facenet-webface_bfw_embeddings.pk', 'rb'))[['img_path', 'embedding']]
    given = pd.read_csv('./data/bfw/bfw.csv').rename(columns={'same': 'label'})[['label', 'p1', 'p2', 'facenet']]
    print('Total reference cosine pairs:', len(given))

    # Make sure both dfs have same formatted paths
    embeddings['img_path'] = embeddings['img_path'].apply(
        lambda x: x.replace('\\', '/').replace('data/bfw/bfw-cropped-aligned/', '')
    )
    
    # Get only the images for which we have embeddings
    given = given[(given['p1'].isin(embeddings['img_path'])) & (given['p2'].isin(embeddings['img_path']))]
    given = given.sample(20000)
    print(f'Considering {len(given)} pairs')

    # Get cosine similarity
    given['cos_sim'] = given.apply(lambda x: getCosSim(x, embeddings), axis=1)

    # Calculate difference and report metrics
    difference = (given['cos_sim'] - given['facenet']).to_numpy()
    mean = round(difference.mean(), 3)
    std = round(difference.std(), 3)
    corr = round(given[["cos_sim","facenet"]].corr()["cos_sim"]["facenet"], 3)
    
    print("Comparing cosine similarity of photo-pair embeddings")
    print(f'Mean: {mean}')
    print(f'Std: {std}')
    print(f'Linear corr: {corr}')
    
    difference_same = (given[given["label"]==1]['cos_sim'] - given[given["label"]==1]['facenet']).to_numpy()
    difference_diff = (given[given["label"]==0]['cos_sim'] - given[given["label"]==0]['facenet']).to_numpy()
    
    plt.hist(difference_same, bins=40, alpha=0.7)
    plt.hist(difference_diff, bins=40, alpha=0.7)
    plt.plot([],[],' ', label=f'n={len(difference)}\nMean={mean}\nStd={std}\nCorr={corr}')
    plt.xlabel("predicted - reference")
    plt.ylabel("frequency")
    plt.legend(handletextpad=.0, handlelength=0)
    plt.title("Comparison of cosine similarity of photo-pair embeddings")
    plt.show()

def getCosSim(row, embeddings):
    embedding_p1 = embeddings[embeddings['img_path'] == row['p1']]['embedding'].to_numpy()[0]
    embedding_p2 = embeddings[embeddings['img_path'] == row['p2']]['embedding'].to_numpy()[0]
    return cos_sim(embedding_p1, embedding_p2)

def cos_sim(a, b):
    return np.dot(a,b) / (norm(a)*norm(b)) 

if __name__ == '__main__':
    start = time.time()
    # main()
    sim_histograms_with_existing_file("./data/bfw/bfw_w_sims.csv", "arcface")
    sim_histograms("./similarities/arcface_bfw_cosin_sim.csv")
    print(f'Took {round(time.time() - start)} seconds')