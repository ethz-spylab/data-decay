#%%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
#CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings.npy"
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
recompute_kmeans = True

#%%
# Cluster the embeddings with k-means
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import torch
import os

#%%
print(torch.version.cuda)
#%%
# cuda available devices

os.environ['OPENBLAS_NUM_THREADS'] = '1'

torch.cuda.is_available()
device = torch.device('cuda:7')

#%%
# Load the embeddings
cc_embeddings = np.load(CC_EMBEDDINGS_FOLDER)

#%%
print(cc_embeddings.shape)

#%%
CLUSTER_CENTERS = EMBEDDINGS_FOLDER / "cluster_centers_L14_500.npy" # can also be just "cluster_centers.npy"
N_CLUSTERS = 500
kmeans_fitter =  MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=256 * 16, verbose=1, n_init=5, max_iter=500, random_state=0)
print("Starting k-means")
kmeans = kmeans_fitter.fit(cc_embeddings)

print("Saving k-means")
# Save the cluster centers
np.save(CLUSTER_CENTERS, kmeans.cluster_centers_)
cluster_centers = kmeans.cluster_centers_

#%%
# Load the cluster centers
CLUSTER_CENTERS = EMBEDDINGS_FOLDER / "cluster_centers_L14_500.npy" # can also be just "cluster_centers.npy"
cluster_centers = np.load(CLUSTER_CENTERS)


print("Dot products")

#%%
def dot_products_distances(emb_A, emb_B):
    """Compute the dot products between all pairs of vectors in emb_A and emb_B.
    Args:
        emb_A: np.array of shape (n_A, d)
        emb_B: np.array of shape (n_B, d)
    Returns:
        np.array of shape (n_A, n_B) with the dot products.
    """
    import torch
    emb_A = torch.from_numpy(emb_A).to(torch.float32).to(device)
    emb_B = torch.from_numpy(emb_B).to(torch.float32).to(device)
    dot_products = torch.mm(emb_A, emb_B.t()).cpu().numpy()
    distances = torch.cdist(emb_A, emb_B).cpu().numpy()
    return dot_products, distances


dot_products, distances = dot_products_distances(cc_embeddings, cluster_centers)
# print shapes
print("dot_products", dot_products.shape)
print("distances", distances.shape)

#%%
# Save
DOT_PRODUCTS = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_L14_500.npy"
np.save(DOT_PRODUCTS, dot_products)
print("Saved dot products to", DOT_PRODUCTS)

DISTANCES = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_distances_L14_500.npy"
np.save(DISTANCES, distances)
print("Saved distances to", DISTANCES)


#%%
(kmeans.cluster_centers_).shape




# %%
