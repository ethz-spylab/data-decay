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
        self.decayed_indices_path = '/data/cc3m/decayed_indices.txt'
        self.separate_decayed_indices_path = '/data/cc3m/script_tests/decayed_indices/decayed_indices.txt'
        self.clusters_folder = '/data/cc3m/script_tests/clusters/'
        self.decayed_samples_dict_nn_path = '/data/cc3m/script_tests/diclist_nn.json'
        self.decayed_dict_calculate = True
        self.consider_nns = True
        self.similarity_type = 'dot_products'
        self.captions_urls_path = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
        self.result_folder = '/data/cc3m/script_tests/results/'
        self.nearby_sample_count = 20
        self.nearby_decayed_sample_count_threshold = 10
        self.closest_clusters_count = 3
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
""" decayed_indices_path = args.decayed_indices_path
with open(decayed_indices_path, "r") as f:
    decayed_indices = json.load(f) """

decayed_indices_path = args.decayed_indices_path
decayed_indices = []
with open(decayed_indices_path, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))

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




# %%
#Test without other closest clusters
""" print(f'Testing without other closest clusters')
nn_decayed_counts = [diclist_nn[x]['nn_decayed_count'] for x in range(len(diclist_nn))]
check = np.array(nn_decayed_counts) >= args.nearby_decayed_sample_count_threshold

if args.check_similarity:
    nn_score_th = args.lower_similarity_threshold
    nn_scores = [diclist_nn[x]['nn_scores'] for x in range(len(diclist_nn))]
    nn_indices = [diclist_nn[x]['nn_indices'] for x in range(len(diclist_nn))]

    check2 = []
    for i in range(len(nn_scores)):
        nn_decayed_inds_temp = decayed_array[np.array(nn_indices[i])] == 1
        nn_decayed_scores_temp = np.array(nn_scores[i])[nn_decayed_inds_temp]
        if len(nn_decayed_scores_temp) == 0:
            check2.append(False)
        else:
            check2.append(nn_decayed_scores_temp[-1] >= nn_score_th)
    check = check & check2

good_ones = np.where(check)[0]
good_indices = [diclist_nn[x]['decayed_indice'] for x in good_ones]
if args.consider_nns:
    good_indices_neighbours = []
    for i in good_ones:
        good_indices_neighbours.extend(diclist_nn[i]['nn_indices'])
    good_indices.extend(good_indices_neighbours)
    good_indices = np.unique(good_indices).tolist()

print(f'Number of good ones: {len(good_ones)}')
print(f'Number of good indices: {len(good_indices)}')

targeted_decay_indices = np.array([np.array(x) for x in separate_decayed_indices[:-1]])
random_decay_indices = np.array(separate_decayed_indices[-1])
combined_decay_indices = np.array(decayed_indices)

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

tot_targeted_decay_indices = np.unique(np.concatenate(targeted_decay_indices))

sums = [len(x) for x in targeted_decay_groups_significant_assignment]
print(f'Number of samples in each targeted group: \n{sums}')
print(f'Total number of samples in targeted groups (with duplicates): \n{sum(sums)}')
print(f'Total number of samples in targeted groups (without duplicates): \
      \n{len(np.intersect1d(tot_targeted_decay_indices, good_indices))}')

random_decay_indices_significant_assignment = np.intersect1d(random_decay_indices, good_indices)
combined_decay_indices_significant_assignment = np.intersect1d(combined_decay_indices, good_indices)
 """
# %%
""" print(f'Number of assigned random decayed samples without closest clusters: \
      \n{len(random_decay_indices_significant_assignment)}') """
#print(captions[random_decay_indices_significant_assignment])
#k = 8
#print(f'{good_indice_dict[good_indices[8]]}')
#diclist_close_k[decayed_interest_dict[good_indices[8]]]
#print(f'caption: {captions[good_indices[k]]}')
# %%








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
if args.consider_nns:
    print("Considering the neighbours")
    good_indices_neighbours = []
    for i in good_ones:
        good_indices_neighbours.extend(diclist_nn_close_k[i]['nn_indices_close_k'])
    good_indices.extend(good_indices_neighbours)
    good_indices = np.unique(good_indices).tolist()

print(f'Number of good ones: {len(good_ones)}')
print(f'Number of good indices, pre-filter for decayed: {len(good_indices)}')

# remove the non-decayed samples from the good_indices
good_indices = np.array(good_indices)
good_indices = good_indices[decayed_array[good_indices]==1].tolist()

print(f'Number of good indices, post-filter for decayed: {len(good_indices)}')

# %%

targeted_decay_indices = np.array([np.array(x) for x in separate_decayed_indices[:-1]])
random_decay_indices = np.array(separate_decayed_indices[-1])
combined_decay_indices = np.array(decayed_indices)

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

tot_targeted_decay_indices = np.unique(np.concatenate(targeted_decay_indices))

sums = [len(x) for x in targeted_decay_groups_significant_assignment]
print(f'Number of samples in each targeted group: \n{sums}')
print(f'Total number of samples in targeted groups (with duplicates): \n{sum(sums)}')
print(f'Total number of samples in targeted groups (without duplicates): \
      \n{len(np.intersect1d(tot_targeted_decay_indices, good_indices))}')

random_decay_indices_significant_assignment = np.intersect1d(random_decay_indices, good_indices)
combined_decay_indices_significant_assignment = np.intersect1d(combined_decay_indices, good_indices)

print(f'Number of assigned random decayed samples with closest clusters: \
      \n{len(random_decay_indices_significant_assignment)}')
# %%


# Find a way to deal with repeated ones


# %%
""" print(f'Number of assigned random decayed samples with closest clusters: \
      \n{len(random_decay_indices_significant_assignment)}') """
#print(captions[random_decay_indices_significant_assignment])
# %%
#k = 8
#print(f'{good_indice_dict[good_indices[8]]}')
#diclist_close_k[decayed_interest_dict[good_indices[8]]]
#print(f'caption: {captions[good_indices[k]]}')
# %%
#print(captions[targeted_decay_groups_significant_assignment[9]])
# %%

# print the overlaps between the targeted groups
""" for i in range(len(targeted_decay_groups_significant_assignment)):
    for j in range(i+1, len(targeted_decay_groups_significant_assignment)):
        overlap = np.intersect1d(targeted_decay_groups_significant_assignment[i],
                                    targeted_decay_groups_significant_assignment[j])
        overlap = overlap.tolist()
        if len(overlap) > 0:
            print(f'Overlap between targeted groups {i} and {j}: {overlap}')
        for k in overlap:
            print(f'caption {k}: {captions[k]}') """
# %%
relevant_captions = captions[good_indices]
# %%
""" relevant_dataset_embeddings = dataset_embeddings[good_indices]
n_clusters = 10
kmeans_fitter =  MiniBatchKMeans(n_clusters=n_clusters, batch_size=256 * 2, verbose=0, n_init=5, max_iter=500, random_state=42)
kmeans = kmeans_fitter.fit(relevant_dataset_embeddings)
relevant_cluster_centers = kmeans.cluster_centers_
relevant_dot_products, relevant_distances = dot_products_distances(relevant_dataset_embeddings, relevant_cluster_centers)
relevant_cluster_assignments = np.argmax(relevant_dot_products, axis=1)
relevant_cluster_counts = Counter(relevant_cluster_assignments)

for i in range(n_clusters):
    print(f'# of samples in cluster {i}: {relevant_cluster_counts[i]}')

for i in range(n_clusters):
    print(relevant_captions[relevant_cluster_assignments==i][:10]) """
# %%
relevant_dataset_embeddings = dataset_embeddings[good_indices]
n_clusters = args.num_clusters
km = kmeans_core(k=n_clusters, data_array=relevant_dataset_embeddings, batch_size=256 * 16, epochs=200, all_cuda=True)
print("Starting torch k-means")
km.run()
relevant_cluster_centers = km.cent.cpu().numpy()
# %%
relevant_dot_products, relevant_distances = dot_products_distances(relevant_dataset_embeddings, relevant_cluster_centers)
relevant_cluster_assignments = np.argmax(relevant_dot_products, axis=1)
relevant_cluster_counts = Counter(relevant_cluster_assignments)
# %%
for i in range(n_clusters):
    print(f'# of samples in cluster {i}: {relevant_cluster_counts[i]}')
# %%
for i in range(n_clusters):
    print(relevant_captions[relevant_cluster_assignments==i][:10])
# %%
cluster_captions = [relevant_captions[relevant_cluster_assignments==i].tolist() for i in range(n_clusters)]
# %%
# save the cluster captions to a json file
cluster_captions_path = os.path.join(args.result_folder, 'cluster_captions.json')
with open(cluster_captions_path, 'w') as fout:
    json.dump(cluster_captions, fout)
# %%
""" for i in range(n_clusters):
    cluster_captions_path = os.path.join(args.result_folder, f'cluster_captions_{i}.txt')
    with open(cluster_captions_path, 'w') as fout:
        fout.write(cluster_captions[i]) """
# %%
