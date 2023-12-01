from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import torch
import os
import argparse

from utils import dot_products_distances

parser = argparse.ArgumentParser(description='Get clusters using k-means and find the similarities and distances of samples to clusters centers.')
parser.add_argument('--embeddings_path', type=str, default='.embeddings/text_embeddings.npy',
                    help='embeddings path')
parser.add_argument('--cluster_count', type=int, default=100,
                    help='cluster count')
parser.add_argument('--save_folder', type=str, default='clusters',
                    help='save folder')
parser.add_argument('--cuda_device', type=int, default=-1, 
                    help='cuda device')
parser.add_argument('--verbose', type=int, default=0,
                    help='verbosity of kmeans fitter')
parser.add_argument('--torch', action='store_true',
                    help='use torch kmeans instead of scikit-learn minibatch kmeans')

args = parser.parse_args()

os.environ['OPENBLAS_NUM_THREADS'] = '1'
cuda_device = "cuda:{}".format(args.cuda_device)
device = torch.device(cuda_device)

# Load the embeddings
embeddings_path = args.embeddings_path
embeddings = np.load(embeddings_path)

# Find the clusters
cluster_count = args.cluster_count
verbose = args.verbose
if args.torch:
    from kmean_torch import kmeans_core
    #km = kmeans_core(k=K,data_array=A,batch_size=3000,epochs=200)
    km = kmeans_core(k=cluster_count, data_array=embeddings, batch_size=256 * 16, epochs=200, all_cuda=True)
    print("Starting torch k-means")
    km.run()
    cluster_centers = km.cent.cpu().numpy()

else:
    kmeans_fitter =  MiniBatchKMeans(n_clusters=cluster_count, batch_size=256 * 16, verbose=verbose, n_init=5, max_iter=500, random_state=42)
    print("Starting k-means")
    kmeans = kmeans_fitter.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

# Save the cluster centers
save_folder = args.save_folder
if not os.path.exists(save_folder):
    print(f'Creating folder {save_folder}')
    os.makedirs(save_folder)
cluster_centers_path = os.path.join(save_folder, 'cluster_centers.npy')
np.save(cluster_centers_path, cluster_centers)

# Find the similarities and distances of samples to clusters centers
print("Calculating dot products and distances")
dot_products , distances = dot_products_distances(embeddings, cluster_centers)

# Save the similarities and distances
print("Saving dot products and distances")
dot_products_path = os.path.join(save_folder, 'dot_products.npy')
distances_path = os.path.join(save_folder, 'distances.npy')
np.save(dot_products_path, dot_products)
np.save(distances_path, distances)

