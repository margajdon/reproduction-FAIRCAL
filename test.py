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
    axs[0].set_title(f"Distribution of cos_sim ({filename.split('/')[-1]} - {model})")

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

def getCosSim(row, embeddings):
    embedding_p1 = embeddings[embeddings['img_path'] == row['p1']]['embedding'].to_numpy()[0]
    embedding_p2 = embeddings[embeddings['img_path'] == row['p2']]['embedding'].to_numpy()[0]
    return cos_sim(embedding_p1, embedding_p2)

def cos_sim(a, b):
    return np.dot(a,b) / (norm(a)*norm(b)) 

if __name__ == '__main__':
    start = time.time()
    sim_histograms_with_existing_file("./data/bfw/bfw_w_sims.csv", "arcface")
    sim_histograms_with_existing_file("./data/bfw/bfw_w_sims.csv", "facenet-webface")

    sim_histograms_with_existing_file("./data/rfw/rfw_w_sims.csv", "facenet")
    sim_histograms_with_existing_file("./data/rfw/rfw_w_sims.csv", "facenet-webface")

    print(f'Took {round(time.time() - start)} seconds')