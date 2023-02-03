#!/usr/bin/env python
# coding: utf-8

# # Cluster Visualisations for FairCal based on Salvador et al., 2022

# ### Imports

# In[ ]:


import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ### Loading a cluster file

# In[ ]:


# def load_clusters(dataset, feature, n_clusters, fold_nr, clustering_method):
#     clustering_method = 'clustering_faircal'
#     filename = f'experiments/{clustering_method}/{dataset}_{feature}_nclusters{str(n_clusters)}_fold{str(fold_nr)}'
#     results = np.load(f'{filename}.npy', allow_pickle=True).item()
    
#     return results


# In[ ]:


def load_clusters(dataset, feature, n_clusters, fold_nr, clustering_method):
    folder_name = None
    if clustering_method == 'kmeans':
        folder_name = 'clustering_faircal'
    elif clustering_method == 'kmeans-fsn':
        folder_name = 'clustering_fsn'
    elif clustering_method == 'gmm-discrete':
        folder_name = 'clustering_faircal-gmm'
    else:
        print('Please provide a valid clustering method.')  
    filename = f'experiments/{folder_name}/{dataset}_{feature}_nclusters{str(n_clusters)}_fold{str(fold_nr)}'
    results = np.load(f'{filename}.npy', allow_pickle=True).item()
    return results


# ### Example dataset-feature-cluster-fold-clustering technique combination with K-means

# In[ ]:


dataset = 'rfw'
feature = 'facenet-webface'
n_clusters = 100
fold_nr = 1
clustering_method = 'gmm-discrete'

img_array = load_clusters(dataset, feature, n_clusters, fold_nr, clustering_method)


# ### Loading the embeddings

# In[ ]:


def load_embeddings(dataset, feature):
    embedding_data = pickle.load(open(f'embeddings/{feature}_{dataset}_embeddings.pk', 'rb'))
    
    return embedding_data


# In[ ]:


embedding_data = load_embeddings(dataset, feature)


# ### Predicting clusters

# In[ ]:


img_array.trainer_params['accelerator'] = None
img_array.trainer_params_user['accelerator'] = None

# centroids = img_array.model_.centroids.numpy()
# images = np.vstack(embedding_data['embedding'].to_numpy()).astype('double')
# def predict(x, c): np.argmax([np.linalg.norm(x, c_) for c_ in c])

# embedding_data['i_cluster'] = [predict(x, centroids) for x in np.vstack(embedding_data['embedding'].to_numpy()).astype('double')]

embedding_data['i_cluster'] = img_array.predict(np.vstack(embedding_data['embedding'].to_numpy()))


# ### Storing information on what image paths belong to a each cluster

# In[ ]:


cluster_scores = {}

for cluster_id in range(n_clusters):
    cond = embedding_data['i_cluster'] == cluster_id
    cluster_scores[str(cluster_id + 1)] = embedding_data['img_path'][cond].tolist()


# ### Visualising a sample cluster obtained by K-means

# In[ ]:


def visualise_clusters(cluster_scores, start, stop, dataset, fold_nr, feature):
    x_size = 10
    y_size = 10
    clusters_to_visualise = [x for x in range(start, stop)]

    f, axarr = plt.subplots(x_size, y_size)

    number_plotted_already = 0

    for cluster_to_visualise in clusters_to_visualise:
        number_plotted_already = 0
        for x_axis in range(x_size):
            for y_axis in range(y_size):
                img_path = cluster_scores[str(cluster_to_visualise)][number_plotted_already]
                image = mpimg.imread(img_path)
                axarr[x_axis, y_axis].margins(x=0)
                axarr[x_axis, y_axis].imshow(image)
                axarr[x_axis, y_axis].margins(x=0)
                axarr[x_axis, y_axis].axis('off')
                number_plotted_already += 1

        f.suptitle(f"Cluster {cluster_to_visualise} \n{dataset}, fold {fold_nr}, fitted on {feature}")
        f.savefig(f"Cluster{cluster_to_visualise}.pdf", bbox_inches = 'tight')
  


# In[ ]:


visualise_clusters(cluster_scores, 1, 6, dataset, fold_nr, feature)


# In[ ]:


x_size = 10
y_size = 10
cluster_to_visualise = 1

f, axarr = plt.subplots(x_size, y_size)
# f.subplots_adjust(left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.2,
#                     hspace=0.2)
# f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)


number_plotted_already = 0
for x_axis in range(x_size):
    for y_axis in range(y_size):
        img_path = cluster_scores[str(cluster_to_visualise)][number_plotted_already]
        image = mpimg.imread(img_path)
        axarr[x_axis, y_axis].margins(x=0)
        axarr[x_axis, y_axis].imshow(image)
        axarr[x_axis, y_axis].margins(x=0)
        axarr[x_axis, y_axis].axis('off')
        number_plotted_already += 1


    f.savefig(f"Cluster{cluster_to_visualise}.pdf", bbox_inches = 'tight', pad_inches = 0.2)


# ### Example dataset-feature-cluster-fold-clustering technique combination with GMMs

# In[ ]:


dataset = 'rfw'
feature = 'facenet'
n_clusters = 100
fold_nr = 5
clustering_method = 'gmm-discrete'

img_array = load_clusters(dataset, feature, n_clusters, fold_nr, clustering_method)


# In[ ]:


embedding_data = load_embeddings(dataset, feature)


# In[ ]:

img_array.trainer_params['accelerator'] = None
img_array.trainer_params_user['accelerator'] = None
embedding_data['i_cluster'] = img_array.predict(np.vstack(embedding_data['embedding'].to_numpy()))


# In[ ]:


cluster_scores = {}

for cluster_id in range(n_clusters):
    cond = embedding_data['i_cluster'] == cluster_id
    cluster_scores[str(cluster_id + 1)] = embedding_data['img_path'][cond].tolist()


# In[ ]:


x_size = 10
y_size = 10
clusters_to_visualise = [x for x in range(1, 6)]

f, axarr = plt.subplots(x_size, y_size)

number_plotted_already = 0

for cluster_to_visualise in clusters_to_visualise:
    number_plotted_already = 0
    for x_axis in range(x_size):
        for y_axis in range(y_size):
            img_path = cluster_scores[str(cluster_to_visualise)][number_plotted_already]
            image = mpimg.imread(img_path)
            axarr[x_axis, y_axis].margins(x=0)
            axarr[x_axis, y_axis].imshow(image)
            axarr[x_axis, y_axis].margins(x=0)
            axarr[x_axis, y_axis].axis('off')
            number_plotted_already += 1
    
    f.suptitle(f"Cluster {cluster_to_visualise} \n{dataset}, fold {fold_nr}, fitted on {feature}")
    f.savefig(f"Cluster{cluster_to_visualise}.pdf", bbox_inches = 'tight')
        


# In[ ]:


visualise_clusters(cluster_scores, 1, 5, dataset, fold_nr, feature)


# In[ ]:




