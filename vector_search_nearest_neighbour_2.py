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
from kmean_torch import kmeans_core

from utils_new import get_top_n_indices, dot_products_distances
# %%
class Args:
    def __init__(self):
        self.dataset_embeddings_path = "/data/cc3m/cc3m_2023/embeddings/text_embeddings_L14.npy"
        #self.decayed_indices_path = '/data/cc3m/script_tests/decayed_indices/combined_decayed_indices.txt'
        self.decayed_indices_path = '/data/cc3m/decayed_indices.json'
        self.separate_decayed_indices_path = '/data/cc3m/script_tests/decayed_indices/decayed_indices.txt'
        self.clusters_folder = '/data/cc3m/script_tests/clusters/'
        self.decayed_samples_dict_nn_path = '/data/cc3m/script_tests/diclist_nn.json'
        self.decayed_dict_calculate = False
        self.consider_nns = True
        self.similarity_type = 'dot_products'
        self.captions_urls_path = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
        self.result_folder = '/data/cc3m/script_tests/results/'
        self.nearby_sample_count = 20
        self.nearby_decayed_sample_count_threshold = 16
        self.closest_clusters_count = 0
        self.check_similarity = True
        self.lower_similarity_threshold = 0.8
        self.num_clusters = 50
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

""" decayed_indices_path = args.decayed_indices_path
decayed_indices = []
with open(decayed_indices_path, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip())) """

decayed_indices_size = len(decayed_indices)
if args.verbose:
    print(f'Number of decayed indices: {decayed_indices_size}')

decayed_array = np.zeros(dataset_size)
decayed_array[decayed_indices] = 1

separate_decayed_indices_path = args.separate_decayed_indices_path
with open(separate_decayed_indices_path, "r") as f:
    separate_decayed_indices = json.load(f)

targeted_decay_indices = np.array([np.array(x) for x in separate_decayed_indices[:-1]])
random_decay_indices = np.array(separate_decayed_indices[-1])
combined_decay_indices = np.array(decayed_indices)

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
# For each decayed sample, find the closest n samples (decayed or not) in the same cluster

if os.path.exists(args.decayed_samples_dict_nn_path) and not args.decayed_dict_calculate:
    print(f'Loading decayed samples dict nn from {args.decayed_samples_dict_nn_path}')
    with open(args.decayed_samples_dict_nn_path) as fp:
        diclist_nn = json.load(fp)
else:
    print(f'Creating decayed samples dict nn at {args.decayed_samples_dict_nn_path}')

    # For each cluster find the samples assigned to it
    clusters_all = []
    clusters_decayed = []
    for i in range(cluster_counts):
        clusters_all.append(np.where((cluster_assignment == i))[0])
        clusters_decayed.append(np.where((cluster_assignment == i) & (decayed_array == 1))[0])

    nearby_sample_count = args.nearby_sample_count
    # For each decayed sample, find the closest n samples (decayed or not) 
    diclist_nn = [{'decayed_indice':x,
                'nn_indices':None,
                'nn_scores':None,
                'nn_decayed_count':None} for x in decayed_indices]

    # Realize that similar_inds = get_top_n_indices(temp[j], nearby_sample_count) will include
    # the decayed samples as well. So we add 1 to the nearby_sample_count and then remove the first
    for i in tqdm(range(cluster_counts)):
        temp = dataset_embeddings[clusters_decayed[i]] @ dataset_embeddings[clusters_all[i]].T
        for j in range(len(clusters_decayed[i])):
            decayed_ind = clusters_decayed[i][j]
            similar_inds = get_top_n_indices(temp[j], nearby_sample_count + 1)[1:]
            nn_decayed_inds = clusters_all[i][similar_inds]
            diclist_nn[decayed_dict[decayed_ind]]['nn_indices'] = nn_decayed_inds.tolist()
            diclist_nn[decayed_dict[decayed_ind]]['nn_decayed_count'] = int(decayed_array[nn_decayed_inds].sum())
            diclist_nn[decayed_dict[decayed_ind]]['nn_scores'] = temp[j][similar_inds].tolist()

    with open(args.decayed_samples_dict_nn_path, 'w') as fout:
        json.dump(diclist_nn, fout)



# %%
# now include the closest clusters

nn_decayed_counts = [diclist_nn[x]['nn_decayed_count'] for x in range(len(diclist_nn))]
check = np.array(nn_decayed_counts) >= args.nearby_decayed_sample_count_threshold - 2
decayed_of_interest = np.where(check)[0]
decayed_of_interest = [diclist_nn[x]['decayed_indice'] for x in decayed_of_interest]
decayed_of_interest = np.array(decayed_of_interest)

if args.verbose:
    print(f'Start looking at closest {args.closest_clusters_count} clusters to the decayed samples of interest')

diclist_nn_close_k = [{'decayed_indice':x,
                'nn_indices_close_k':diclist_nn[decayed_dict[x]]['nn_indices'].copy(),
                'nn_scores_close_k':diclist_nn[decayed_dict[x]]['nn_scores'].copy(),
                'nn_decayed_count_close_k':None} for x in decayed_of_interest]

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

# For each cluster find the samples assigned to it
clusters_all = []
for i in range(cluster_counts):
    clusters_all.append(np.where((cluster_assignment == i))[0])

# For each decayed sample, find the closest n samples (decayed or not)
nearby_sample_count = args.nearby_sample_count
for i in tqdm(range(cluster_counts)):
    temp = dataset_embeddings[clusters_decayed_of_interest[i]] @ dataset_embeddings[clusters_all[i]].T

    # we no longer need to remove the first element since our decayed samples are not included
    for j in range(len(clusters_decayed_of_interest[i])):
        decayed_ind = clusters_decayed_of_interest[i][j]
        similar_inds = get_top_n_indices(temp[j], nearby_sample_count)
        nn_decayed_inds = clusters_all[i][similar_inds]
        diclist_nn_close_k[decayed_interest_dict[decayed_ind]]['nn_indices_close_k'].extend(nn_decayed_inds.tolist())
        diclist_nn_close_k[decayed_interest_dict[decayed_ind]]['nn_scores_close_k'].extend(temp[j][similar_inds].tolist())

# For each decayed sample of interest, we have (top_k + 1)*nearby_sample_count similar samples.
# find top nearby_sample_count among them
for i in tqdm(range(len(decayed_of_interest))):
    nn_indices_close_k = np.array(diclist_nn_close_k[i]['nn_indices_close_k'])
    nn_scores_close_k = np.array(diclist_nn_close_k[i]['nn_scores_close_k'])
    temp_inds = get_top_n_indices(nn_scores_close_k, nearby_sample_count)
    nn_temp_inds = nn_indices_close_k[temp_inds]
    diclist_nn_close_k[i]['nn_indices_close_k'] = nn_temp_inds.tolist()
    diclist_nn_close_k[i]['nn_scores_close_k'] = nn_scores_close_k[temp_inds].tolist()
    diclist_nn_close_k[i]['nn_decayed_count_close_k'] = int(decayed_array[nn_temp_inds].sum())

nn_decayed_counts = [diclist_nn_close_k[x]['nn_decayed_count_close_k'] for x in range(len(diclist_nn_close_k))]

# %%
check = np.array(nn_decayed_counts) >= args.nearby_decayed_sample_count_threshold

if args.check_similarity:
    nn_score_th = args.lower_similarity_threshold
    nn_scores_close_k = [diclist_nn_close_k[x]['nn_scores_close_k'] for x in range(len(diclist_nn_close_k))]
    nn_indices_close_k = [diclist_nn_close_k[x]['nn_indices_close_k'] for x in range(len(diclist_nn_close_k))]

    check2 = []
    for i in range(len(nn_scores_close_k)):
        nn_decayed_inds_temp = decayed_array[np.array(nn_indices_close_k[i])]==1
        nn_decayed_scores_temp = np.array(nn_scores_close_k[i])[nn_decayed_inds_temp]
        if len(nn_decayed_scores_temp) == 0:
            check2.append(False)
        else:
            check2.append(nn_decayed_scores_temp[-1] >= nn_score_th)
    check = check & check2

good_ones = np.where(check)[0]
good_indices = [diclist_nn_close_k[x]['decayed_indice'] for x in good_ones]

print(f'Number of good ones: {len(good_ones)}')
print(f'Number of good indices, pre-filter for decayed: {len(good_indices)}')

# remove the non-decayed samples from the good_indices
good_indices = np.array(good_indices)
good_indices = good_indices[decayed_array[good_indices]==1].tolist()

print(f'Number of good indices, post-filter for decayed: {len(good_indices)}')
# %%
diclist_good_ones = [diclist_nn_close_k[x].copy() for x in good_ones]
# %%
for i in range(len(diclist_good_ones)):
    diclist_good_ones[i]['decayed_nn_indices'] = [x for x in diclist_good_ones[i]['nn_indices_close_k'] if decayed_array[x]==1]
# %%
good_indices_dict = {}
for i in range(len(good_indices)):
    good_indices_dict[good_indices[i]] = i
# %%
# The problem is sometimes while one sample is in the nn_indices_close_k of another, the other is not in the nn_indices_close_k of the first
clusters = np.ones(len(good_indices), dtype=int)*-1
cluster_counter = 0
for i in range(len(good_indices)):
    if clusters[i] != -1:
        continue
    clusters[i] = cluster_counter
    to_dos = []
    to_dos.extend(diclist_good_ones[i]['decayed_nn_indices'])
    while len(to_dos) > 0:
        temp_indice = to_dos.pop()
        if temp_indice in good_indices_dict:
            temp_indice_pos = good_indices_dict[temp_indice]
            if clusters[temp_indice_pos] == -1:
                clusters[temp_indice_pos] = cluster_counter
                to_dos.extend(diclist_good_ones[temp_indice_pos]['decayed_nn_indices'])
            elif clusters[temp_indice_pos] != cluster_counter:
                old_cluster_count = clusters[temp_indice_pos]
                clusters[clusters==old_cluster_count] = cluster_counter
    
    cluster_counter += 1
# %%
# if two good decayed samples are connected via their decayed_nn_indices, they are in the same cluster
num_clusters = len(np.unique(clusters))
cluster_equivalents = [[] for _ in range(num_clusters)]
nn_to_good_indices = {}
for i in range(len(good_indices)):
    to_dos = diclist_good_ones[i]['decayed_nn_indices']
    for j in range(len(to_dos)):
        if to_dos[j] in nn_to_good_indices:
            if clusters[i] not in nn_to_good_indices[to_dos[j]]:
                nn_to_good_indices[to_dos[j]].append(clusters[i])
        else:
            nn_to_good_indices[to_dos[j]] = [clusters[i]]
# %%
# order nn_to_good_indices by the length of its values
nn_to_good_indices = {k: v for k, v in sorted(nn_to_good_indices.items(), key=lambda item: len(item[1]), reverse=True)}
# %%
matches = [v for k, v in nn_to_good_indices.items() if len(v) > 1]
for match in matches:
    for j in range(len(match)-1):
        clusters[clusters==match[j]] = match[j+1]
# %%
unique_clusters = np.unique(clusters)
num_clusters_2 = len(unique_clusters)
# %%
clusters
# %%
# find the cluster centers and combine the clusters that are too close to each other
# unique_clusters is already sorted
good_dataset_embeddings = dataset_embeddings[np.array(good_indices)]
cluster_centers = np.zeros((num_clusters_2, good_dataset_embeddings.shape[1]))
for i in range(num_clusters_2):
    cluster_centers[i] = np.average(good_dataset_embeddings[clusters == unique_clusters[i]], axis=0)
    cluster_centers[i] = cluster_centers[i]/np.linalg.norm(cluster_centers[i])
# %%
# find the cosine similarity between the cluster centers. put 0 for the diagonal
# put 0's for upper triangular part
cluster_similarity = cluster_centers @ cluster_centers.T
np.fill_diagonal(cluster_similarity, 0)
cluster_similarity = np.tril(cluster_similarity)
# %%
rows, cols = np.where(cluster_similarity > 0.8)
print(len(np.unique(rows)))
# %%
captions[np.array(good_indices)[clusters==unique_clusters[rows[43]]][:10]]
# %%
captions[np.array(good_indices)[clusters==unique_clusters[cols[43]]][:10]]
# %%
cluster_similartity_threshold = 0.8
for i in range(num_clusters_2 - 1, 0, -1):
    temp_sim = cluster_similarity[i,:]
    am = np.argmax(temp_sim)
    if temp_sim[am] > cluster_similartity_threshold:
        clusters[clusters==unique_clusters[i]] = unique_clusters[am]
# %%
counter = Counter(clusters)
# %%
counter
# %%
# look at ones in counter with count > 10
len([x for x in counter.items() if x[1] > 20])
# %%
counter.most_common(20)
# %%
captions[np.array(good_indices)[clusters==29775][:10]]
# %%
# find the number of unique values in the clusters
len(np.unique(clusters))
# %%
np.array(good_indices)[clusters==23].tolist()
# %%
diclist_good_ones[good_indices_dict[np.array(good_indices)[clusters==999].tolist()[0]]]
# %%
diclist_good_ones[good_indices_dict[1763701]]
# %%
print(f'55608: {clusters[good_indices_dict[55608]]}')
clusters[good_indices_dict[1763701]]
# %%
good_indices_dict[1763701]
# %%
if 1763701 in good_indices_dict:
    print('yes')
# %%
diclist_good_ones[0]
# %%
# find neighbors for each decayed element
all_neighbors_dict = {}
for i in tqdm(range(len(diclist_good_ones))):
    all_neighbors_dict[good_indices[i]] = diclist_good_ones[i]['decayed_nn_indices']
    for nearby in diclist_good_ones[i]['decayed_nn_indices']:
        if nearby in all_neighbors_dict:
            all_neighbors_dict[nearby].append(good_indices[i])
        else:
            all_neighbors_dict[nearby] = [good_indices[i]]
# %%
all_neighbors_dict
# %%
print("hi")
# %%
np.array(good_indices)[clusters==29501]
# %%
diclist_nn_close_k[decayed_interest_dict[3283665]]['nn_indices_close_k']
# %%
captions[diclist_nn_close_k[decayed_interest_dict[3283665]]['nn_indices_close_k']]
# %%
116762 in good_indices
# %%
for i in range(3, 0, -1):
    print(i)
# %%
