# %%
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

from tqdm import tqdm
import json
import random
import argparse

from utils_new import get_top_n_indices, dot_products_distances
# %%
class Args:
    def __init__(self):
        self.dataset_embeddings_path = "/data/cc3m/cc3m_2023/embeddings/text_embeddings_L14.npy"
        self.decayed_indices_path = '/data/cc3m/script_tests/decayed_indices/combined_decayed_indices.txt'
        self.clusters_folder = '/data/cc3m/script_tests/clusters/'
        self.decayed_samples_dict_path = '/data/cc3m/script_tests/diclist.json'
        self.decayed_dict_calculate = False
        self.similarity_type = 'dot_products'
        self.captions_urls_path = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
        self.result_folder = '/data/cc3m/script_tests/results/'
        self.nearby_sample_count_threshold_exist = 10
        self.nearby_sample_count_threshold_decayed = 10
        self.closest_clusters_count_exist = 3
        self.closest_clusters_count_decayed = 10
        self.similarity_to_existing_samples_threshold = 0.6
        self.similarity_to_decayed_samples_lower_threshold = 0.6
        self.similarity_to_decayed_samples_upper_threshold = 1.0
        self.similar_decayed_samples_count_threshold = 10
        self.decayed_sample_clustering = True
        self.number_of_decayed_sample_clusters = 5
        self.verbose = 1

args = Args()
# %%
captions_urls_path = args.captions_urls_path
captions_urls = pd.read_csv(captions_urls_path, sep="\t", header=None)
captions_urls.columns = ["caption", "url"]
captions = np.array(captions_urls["caption"])
url = np.array(captions_urls["url"])

# %%
# Load the dataset embeddings
if args.verbose:
    print("Loading dataset embeddings")
dataset_embeddings_path = args.dataset_embeddings_path
dataset_embeddings = np.load(dataset_embeddings_path)

dataset_size = dataset_embeddings.shape[0]
if args.verbose:
    print(f'Number of dataset samples: {dataset_size}')

# Load the list of decayed indices
decayed_indices_path = args.decayed_indices_path
with open(decayed_indices_path, "r") as f:
    decayed_indices = json.load(f)

decayed_indices_size = len(decayed_indices)
if args.verbose:
    print(f'Number of decayed indices: {decayed_indices_size}')

decayed_array = np.zeros(dataset_size)
decayed_array[decayed_indices] = 1

# Load the cluster centers, distances, and similarities
if args.verbose:
    print("Loading cluster centers.")
cluster_centers_path = os.path.join(args.clusters_folder, 'cluster_centers.npy')
cluster_centers = np.load(cluster_centers_path)
if args.verbose:
    print("Loading distances")
distances_path = os.path.join(args.clusters_folder, 'distances.npy')
distances = np.load(distances_path)
if args.verbose:
    print("Loading similarities")
dot_products_path = os.path.join(args.clusters_folder, 'dot_products.npy')
dot_products = np.load(dot_products_path)


if args.similarity_type == 'distances':
    similarity = 1 - distances
elif args.similarity_type == 'dot_products':
    similarity = dot_products
else:
    raise ValueError("Similarity type should be either distances or dot_products.")
# Find the cluster assignments using argmax
cluster_assignment = np.argmax(similarity, axis=1)

# %%
cluster_counts = len(cluster_centers)
decayed_dict = {}
for i in range(len(decayed_indices)):
    decayed_dict[decayed_indices[i]] = i
# %%
if os.path.exists(args.decayed_samples_dict_path) and not args.decayed_dict_calculate:
    print(f'Loading decayed samples dict from {args.decayed_samples_dict_path}')
    with open(args.decayed_samples_dict_path) as fp:
        diclist = json.load(fp)
else:
    print(f'Creating decayed samples dict at {args.decayed_samples_dict_path}')

    exist_nearby_count = args.nearby_sample_count_threshold_exist
    decayed_nearby_count = args.nearby_sample_count_threshold_decayed

    clusters_decayed = []
    clusters_exist = []
    for i in range(cluster_counts):
        clusters_decayed.append(np.where((cluster_assignment == i) & (decayed_array == 1))[0])
        clusters_exist.append(np.where((cluster_assignment == i) & (decayed_array == 0))[0])

    diclist = [{'decayed_indice':x,
                'similar_decayed':None,
                'decayed_scores':None,
                'similar_exist':None,
                'exist_scores':None} for x in decayed_indices]
        
    for i in tqdm(range(cluster_counts)):
        temp_exist = dataset_embeddings[clusters_decayed[i]] @ dataset_embeddings[clusters_exist[i]].T
        temp_decayed = dataset_embeddings[clusters_decayed[i]] @ dataset_embeddings[clusters_decayed[i]].T
        # set the diagonal to 0 for the decayed samples
        np.fill_diagonal(temp_decayed, 0)
        for j in range(len(clusters_decayed[i])):
            decayed_ind = clusters_decayed[i][j]
            similar_decayed_inds = get_top_n_indices(temp_decayed[j], decayed_nearby_count)
            diclist[decayed_dict[decayed_ind]]['similar_decayed'] = clusters_decayed[i][similar_decayed_inds].tolist()
            diclist[decayed_dict[decayed_ind]]['decayed_scores'] = temp_decayed[j][similar_decayed_inds].tolist()

            similar_exist_inds = get_top_n_indices(temp_exist[j], exist_nearby_count)
            diclist[decayed_dict[decayed_ind]]['similar_exist'] = clusters_exist[i][similar_exist_inds].tolist()
            diclist[decayed_dict[decayed_ind]]['exist_scores'] = temp_exist[j][similar_exist_inds].tolist()
    
    with open(args.decayed_samples_dict_path, 'w') as fout:
        json.dump(diclist, fout)
# %%
exist_nearby_count = args.nearby_sample_count_threshold_exist
decayed_nearby_count = args.nearby_sample_count_threshold_decayed

""" similarity_to_existing_samples_threshold = args.similarity_to_existing_samples_threshold
similarity_to_decayed_samples_lower_threshold = args.similarity_to_decayed_samples_lower_threshold """

similarity_to_existing_samples_threshold = args.similarity_to_existing_samples_threshold
similarity_to_decayed_samples_lower_threshold = args.similarity_to_decayed_samples_lower_threshold

existing_similarities = [diclist[x]['exist_scores'][exist_nearby_count - 1] for x in range(len(diclist))]
decayed_similarities = [diclist[x]['decayed_scores'][decayed_nearby_count - 1] for x in range(len(diclist))]
# %%
existing_check = np.array(existing_similarities) < similarity_to_existing_samples_threshold
decayed_check = np.array(decayed_similarities) > similarity_to_decayed_samples_lower_threshold
# %%
good_indices = np.where(existing_check & decayed_check)[0]
# %%
good_indices = [diclist[x]['decayed_indice'] for x in good_indices]
len(good_indices)
# %%
#captions[good_indices]
# %%
combined_decay_indices = np.array(decayed_indices)
# %%
good_indice_dict = {}
for i in range(len(good_indices)):
    good_indice_dict[good_indices[i]] = {'text_assigned':None}
combined_decay_indices_significant_assignment = np.intersect1d(combined_decay_indices, good_indices)
# %%
