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
        self.separate_decayed_indices_path = '/data/cc3m/script_tests/decayed_indices/decayed_indices.txt'
        self.clusters_folder = '/data/cc3m/script_tests/clusters/'
        self.decayed_samples_dict_path = '/data/cc3m/script_tests/diclist.json'
        self.decayed_dict_calculate = False
        self.similarity_type = 'dot_products'
        self.captions_urls_path = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
        self.result_folder = '/data/cc3m/script_tests/results/'
        self.closest_clusters_count = 3
        self.similarity_to_existing_samples_threshold = 0.8
        self.similarity_to_decayed_samples_lower_threshold = 0.8
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

separate_decayed_indices_path = args.separate_decayed_indices_path
with open(separate_decayed_indices_path, "r") as f:
    separate_decayed_indices = json.load(f)

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

    # For each cluster find the samples assigned to it
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
            similar_decayed_inds = get_top_n_indices(temp_decayed[j], 10)
            diclist[decayed_dict[decayed_ind]]['similar_decayed'] = clusters_decayed[i][similar_decayed_inds].tolist()
            diclist[decayed_dict[decayed_ind]]['decayed_scores'] = temp_decayed[j][similar_decayed_inds].tolist()

            similar_exist_inds = get_top_n_indices(temp_exist[j], 10)
            diclist[decayed_dict[decayed_ind]]['similar_exist'] = clusters_exist[i][similar_exist_inds].tolist()
            diclist[decayed_dict[decayed_ind]]['exist_scores'] = temp_exist[j][similar_exist_inds].tolist()
    
    with open(args.decayed_samples_dict_path, 'w') as fout:
        json.dump(diclist, fout)

# %%
decayed_dict = {}
for i in range(len(decayed_indices)):
    decayed_dict[decayed_indices[i]] = i
# %%
# Looking at the samples only in the same cluster might be misleading. So instead we should look at the other top_k cluster that is most similar
# to the decayed sample. 

similarity_to_existing_samples_threshold = args.similarity_to_existing_samples_threshold
similarity_to_decayed_samples_lower_threshold = args.similarity_to_decayed_samples_lower_threshold

existing_similarities = [diclist[x]['exist_scores'][-1] for x in range(len(diclist))]
decayed_similarities = [diclist[x]['decayed_scores'][-1] for x in range(len(diclist))]

existing_check = np.array(existing_similarities) < similarity_to_existing_samples_threshold
decayed_check = np.array(decayed_similarities) > similarity_to_decayed_samples_lower_threshold

#print(f'# decayed indices with similarity to existing samples less than {similarity_to_existing_samples_threshold}: \
#\nand similarity to decayed samples greater than {similarity_to_decayed_samples_lower_threshold}: \
#{np.sum(existing_check & decayed_check)}')

print(f'# decayed indices with similarity to existing samples less than {similarity_to_existing_samples_threshold}: \
\n{np.sum(existing_check)}')
# %%
decayed_of_interest = np.where(existing_check)[0]
decayed_of_interest = [diclist[x]['decayed_indice'] for x in decayed_of_interest]
decayed_of_interest = np.array(decayed_of_interest)
# %%
if args.verbose:
    print(f'Start looking at closest {args.closest_clusters_count} clusters to the decayed samples of interest')
diclist_close_k = [{'decayed_indice':x,
            'similar_decayed_close_k':diclist[decayed_dict[x]]['similar_decayed'].copy(),
            'decayed_scores_close_k':diclist[decayed_dict[x]]['decayed_scores'].copy(),
            'similar_exist_close_k':diclist[decayed_dict[x]]['similar_exist'].copy(),
            'exist_scores_close_k':diclist[decayed_dict[x]]['exist_scores'].copy()} 
            for x in decayed_of_interest]

decayed_interest_dict = {}
for i in range(len(decayed_of_interest)):
    decayed_interest_dict[decayed_of_interest[i]] = i

# Find the top_k clusters for each decayed sample of interest
close_k = args.closest_clusters_count
decayed_of_interest_clusters = []
for i in range(len(decayed_of_interest)):
    decayed_ind = decayed_of_interest[i]
    decayed_of_interest_clusters.append(get_top_n_indices(similarity[decayed_ind],close_k+1)[1:])

# For each cluster, find the decayed samples(top_k) in that cluster
clusters_decayed_of_interest = [[] for _ in range(cluster_counts)]
for i in range(len(decayed_of_interest)):
    for j in range(close_k):
        clusters_decayed_of_interest[decayed_of_interest_clusters[i][j]].append(decayed_of_interest[i])

clusters_exist = []
clusters_decayed = []
for i in range(cluster_counts):
    clusters_exist.append(np.where((cluster_assignment == i) & (decayed_array == 0))[0])
    clusters_decayed.append(np.where((cluster_assignment == i) & (decayed_array == 1))[0])
# %%
# For each cluster, find the similarity between the decayed samples of interest in that cluster
# and the decayed and the existing samples in that cluster
# For each decayed sample of interest in that cluster, find top 10 similar decayed and existing samples
for i in tqdm(range(cluster_counts)):
    temp_exist = dataset_embeddings[clusters_decayed_of_interest[i]] @ dataset_embeddings[clusters_exist[i]].T
    temp_decayed = dataset_embeddings[clusters_decayed_of_interest[i]] @ dataset_embeddings[clusters_decayed[i]].T

    for j in range(len(clusters_decayed_of_interest[i])):
        decayed_ind = clusters_decayed_of_interest[i][j]

        similar_decayed_inds = get_top_n_indices(temp_decayed[j], 10)
        diclist_close_k[decayed_interest_dict[decayed_ind]]['similar_decayed_close_k'] += clusters_decayed[i][similar_decayed_inds].tolist()
        diclist_close_k[decayed_interest_dict[decayed_ind]]['decayed_scores_close_k'] += temp_decayed[j][similar_decayed_inds].tolist()

        similar_exist_inds = get_top_n_indices(temp_exist[j], 10)
        diclist_close_k[decayed_interest_dict[decayed_ind]]['similar_exist_close_k'] += clusters_exist[i][similar_exist_inds].tolist()
        diclist_close_k[decayed_interest_dict[decayed_ind]]['exist_scores_close_k'] += temp_exist[j][similar_exist_inds].tolist()
# %%
# For each decayed sample of interest, we have (top_k + 1)*10 similar samples. find top 10 among them
for i in tqdm(range(len(decayed_of_interest))):

    exist_scores_close_k = np.array(diclist_close_k[i]['exist_scores_close_k'])
    similar_exist_close_k = np.array(diclist_close_k[i]['similar_exist_close_k'])
    temp_exist_inds = get_top_n_indices(exist_scores_close_k, 10)
    diclist_close_k[i]['exist_scores_close_k'] = exist_scores_close_k[temp_exist_inds].tolist()
    diclist_close_k[i]['similar_exist_close_k'] = similar_exist_close_k[temp_exist_inds].tolist()

    decayed_scores_close_k = np.array(diclist_close_k[i]['decayed_scores_close_k'])
    similar_decayed_close_k = np.array(diclist_close_k[i]['similar_decayed_close_k'])
    temp_decayed_inds = get_top_n_indices(decayed_scores_close_k, 10)
    diclist_close_k[i]['decayed_scores_close_k'] = decayed_scores_close_k[temp_decayed_inds].tolist()
    diclist_close_k[i]['similar_decayed_close_k'] = similar_decayed_close_k[temp_decayed_inds].tolist()
# %%
exist_scores_close_k = [diclist_close_k[x]['exist_scores_close_k'][-1] for x in range(len(diclist_close_k))]
decayed_scores_close_k = [diclist_close_k[x]['decayed_scores_close_k'][2] for x in range(len(diclist_close_k))]

existing_check = np.array(exist_scores_close_k) < similarity_to_existing_samples_threshold
decayed_check = np.array(decayed_scores_close_k) > similarity_to_decayed_samples_lower_threshold
#decayed_check = np.array(decayed_scores_close_k) > 0.7

significant_ones = np.where(existing_check & decayed_check)[0]
significant_ones = [diclist_close_k[x]['decayed_indice'] for x in significant_ones]

print(f'# decayed indices with similarity to existing samples less than {similarity_to_existing_samples_threshold}: \
\nand similarity to decayed samples greater than {similarity_to_decayed_samples_lower_threshold}: \
\n{np.sum(existing_check & decayed_check)}')
# %%
good_ones = np.where(existing_check & decayed_check)[0]
good_indices = [diclist_close_k[x]['decayed_indice'] for x in good_ones]
# %%
print(f'Number of good ones: {len(good_ones)}')
print(f'Number of good indices: {len(good_indices)}')
# %%
targeted_decay_indices = np.array([np.array(x) for x in separate_decayed_indices[:-1]])
random_decay_indices = np.array(separate_decayed_indices[-1])
combined_decay_indices = np.array(decayed_indices)
# %%
good_indice_dict = {}
for i in range(len(good_indices)):
    good_indice_dict[good_indices[i]] = {'text_assigned':None}
targeted_decay_groups_significant_assignment = []
for i in range(len(targeted_decay_indices)):
    targeted_decay_groups_significant_assignment.append(
        np.intersect1d(targeted_decay_indices[i], good_indices)
    )
    for j in targeted_decay_groups_significant_assignment[i]:
        good_indice_dict[j]['text_assigned'] = i
# %%
tot_targeted_decay_indices = np.unique(np.concatenate(targeted_decay_indices))

sums = [len(x) for x in targeted_decay_groups_significant_assignment]
print(f'Number of samples in each targeted group: \n{sums}')
print(f'Total number of samples in targeted groups (with duplicates): \n{sum(sums)}')
print(f'Total number of samples in targeted groups (without duplicates): \
      \n{len(np.intersect1d(tot_targeted_decay_indices, good_indices))}')

random_decay_indices_significant_assignment = np.intersect1d(random_decay_indices, good_indices)
combined_decay_indices_significant_assignment = np.intersect1d(combined_decay_indices, good_indices)

# %%
print(f'Number of assigned random decayed samples with combined checks: \
      \n{len(random_decay_indices_significant_assignment)}')
# %%
#k = 8
#print(f'{good_indice_dict[good_indices[8]]}')
#diclist_close_k[decayed_interest_dict[good_indices[8]]]
#print(f'caption: {captions[good_indices[k]]}')
# %%






# %%
# Now, only do the check for similarity to the existing ones
print(f'Initally get all the decayed ones that satisfy exist threshold, \
      \nthen find the similaries of decayed ones among themselves and use the decay threshold')

good_ones = np.where(existing_check)[0]
good_indices = [diclist_close_k[x]['decayed_indice'] for x in good_ones]
# %%
good_dataset_embeddings = dataset_embeddings[good_indices]
css = good_dataset_embeddings @ good_dataset_embeddings.T
np.fill_diagonal(css, 0)
# %%
css.shape
# %%
lower_th = args.similarity_to_decayed_samples_lower_threshold
count_th = args.similar_decayed_samples_count_threshold
new_good_indices = []
for i in range(len(good_ones)):
    if np.sum(css[i] > lower_th) > count_th:
        new_good_indices.append(good_indices[i])
# %%
print(f'Number of good ones: {len(good_ones)}')
print(f'Number of good indices: {len(good_indices)}')
print(f'Number of new good indices: {len(new_good_indices)}')
# %%
new_good_indice_dict = {}
for i in range(len(new_good_indices)):
    new_good_indice_dict[new_good_indices[i]] = {'text_assigned':None}
new_targeted_decay_groups_significant_assignment = []
for i in range(len(targeted_decay_indices)):
    new_targeted_decay_groups_significant_assignment.append(
        np.intersect1d(targeted_decay_indices[i], new_good_indices)
    )
    for j in np.intersect1d(targeted_decay_indices[i], new_good_indices):
        new_good_indice_dict[j]['text_assigned'] = i
# %%
tot_targeted_decay_indices = np.unique(np.concatenate(targeted_decay_indices))

sums = [len(x) for x in new_targeted_decay_groups_significant_assignment]
print(f'Number of samples in each targeted group: \n{sums}')
print(f'Total number of samples in targeted groups (with duplicates): \n{sum(sums)}')
print(f'Total number of samples in targeted groups (without duplicates): \
      \n{len(np.intersect1d(tot_targeted_decay_indices, new_good_indices))}')

random_decay_indices_significant_assignment = np.intersect1d(random_decay_indices, new_good_indices)
combined_decay_indices_significant_assignment = np.intersect1d(combined_decay_indices, new_good_indices)

# %%
print(f'Number of assigned random decayed samples with separate checks: \
      \n{len(random_decay_indices_significant_assignment)}')
# %%
