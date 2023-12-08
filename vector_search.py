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

from utils import get_top_n_indices, dot_products_distances
# %%
class Args:
    def __init__(self):
        self.dataset_embeddings_path = "/data/cc3m/cc3m_2023/embeddings/text_embeddings_L14.npy"
        self.decayed_indices_path = '/data/cc3m/decayed_indices.txt'
        self.clusters_folder = '/data/cc3m/script_tests/clusters/'
        self.decayed_samples_dict_path = '/data/cc3m/script_tests/diclist.json'
        self.similarity_type = 'dot_products'
        self.captions_urls_path = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
        self.result_folder = '/data/cc3m/script_tests/results/'
        self.closest_clusters_count = 3
        self.similarity_to_existing_samples_threshold = 0.5
        self.similarity_to_decayed_samples_lower_threshold = 0.8
        self.similarity_to_decayed_samples_upper_threshold = 1.0
        self.similar_decayed_samples_count_threshold = 10
        self.decayed_sample_clustering = True
        self.number_of_decayed_sample_clusters = 5
        self.verbose = 1

args = Args()

""" parser = argparse.ArgumentParser(description='Apply vector search to find the decayed samples that are dissimilar to the existing samples while similar to the decayed samples.')

parser.add_argument('--dataset_embeddings_path', type=str, default='embeddings/text_embeddings.npy',
                    help='embeddings path for the dataset. This can be either the image or caption embeddings.')
parser.add_argument('--decayed_indices_path', type=str, default='decayed_indices.txt',
                    help='decayed indices path')
parser.add_argument('--clusters_folder', type=str, default='clusters/',
                    help='clusters folder should include the cluster centers, distances, and similarities.')
parser.add_argument('--decayed_samples_dict_path', type=str, default='diclist.json',
                    help='decayed samples dict path')
parser.add_argument('--similarity_type', type=str, default='dot_products',
                    help='similarity type should be either distances or dot_products.')
parser.add_argument('--captions_urls_path', type=str, default='Train_GCC-training.tsv',
                    help='captions urls path')
parser.add_argument('--result_folder', type=str, default='results/',
                    help='result folder will include the decayed samples and their similar samples. If it does not exist, it will be created.')
parser.add_argument('--closest_clusters_count', type=int, default=3,
                    help='how many other closest clusters to look at for each decayed sample apart from its own cluster')
parser.add_argument('--similarity_to_existing_samples_threshold', type=float, default=0.5,
                    help='if the tenth most similar existing sample to a decayed sample has a similarity score less than this threshold, \
                        then we consider this decayed sample to be dissimilar to the existing samples in its cluster')
parser.add_argument('--similarity_to_decayed_samples_lower_threshold', type=float, default=0.8,
                    help='if a decayed sample has more than this threshold similarity score to another decayed sample, \
                        then we consider this decayed sample to be similar to the other decayed sample')
parser.add_argument('--similarity_to_decayed_samples_upper_threshold', type=float, default=1.0,
                    help='if a decayed sample has more than this threshold similarity score to another decayed sample, \
                        then we consider this decayed sample to be same as the other decayed sample and ignore (repeated case)')
parser.add_argument('--similar_decayed_samples_count_threshold', type=int, default=10,
                    help='how many similar decayed samples a decayed sample should have to be considered')
parser.add_argument('--decayed_sample_clustering', type=bool, default=True,
                    help='whether to cluster the decayed samples or not')
parser.add_argument('--number_of_decayed_sample_clusters', type=int, default=5,
                    help='number of clusters for the decayed samples')
parser.add_argument('--verbose', type=int, default=1,
                    help='verbosity of code')

args = parser.parse_args() """





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
decayed_indices = []
with open(decayed_indices_path, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))

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

if args.similarity_type == 'distances':
    similarity = 1 - distances
elif args.similarity_type == 'dot_products':
    similarity = dot_products
else:
    raise ValueError("Similarity type should be either distances or dot_products.")
# %%
cluster_counts = len(cluster_centers)
decayed_dict = {}
for i in range(len(decayed_indices)):
    decayed_dict[decayed_indices[i]] = i

#diclist holds the decayed indice, indices of 10 most similar existing captions 
#in the same cluster as the decayed indice, and the similarity scores of those captions.
if os.path.exists(args.decayed_samples_dict_path):
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

    diclist = [{'decayed':x,
                'similar_exist':None,
                'similar_scores':None} for x in decayed_indices]

    for i in tqdm(range(cluster_counts)):
        temp = dataset_embeddings[clusters_decayed[i]] @ dataset_embeddings[clusters_exist[i]].T
        for j in range(len(clusters_decayed[i])):
            decayed_ind = clusters_decayed[i][j]
            similar_inds = get_top_n_indices(temp[j], 10)
            diclist[decayed_dict[decayed_ind]]['similar_exist'] = clusters_exist[i][similar_inds].tolist()
            diclist[decayed_dict[decayed_ind]]['similar_scores'] = temp[j][similar_inds].tolist()

    with open(args.decayed_samples_dict_path, 'w') as fout:
        json.dump(diclist, fout)
# %%
decayed_dict = {}
for i in range(len(decayed_indices)):
    decayed_dict[decayed_indices[i]] = i
# %%
similarities = [diclist[i]['similar_scores'][-1] for i in range(len(diclist))]
# %%

# Looking at the samples only in the same cluster might be misleading. So instead we should look at the other top_k cluster that is most similar
# to the decayed sample. Also, we will only do that for the samples we believe might be dissimilar to the existing samples in their cluster.
# Since we are interested in finding decayed samples that are dissimalar to the existing samples, no need to look ones that are similar to the existing samples in their own cluster.

similarity_to_existing_samples_threshold = args.similarity_to_existing_samples_threshold
print(f'% decayed indices with similarity less than {similarity_to_existing_samples_threshold}: \
{np.sum(np.array(similarities) < similarity_to_existing_samples_threshold)/len(similarities)*100:.3f}')
print(f'# decayed indices with similarity less than {similarity_to_existing_samples_threshold}: \
{np.sum(np.array(similarities) < similarity_to_existing_samples_threshold)}')

# %%
decayed_of_interest = np.where(np.array(similarities) < similarity_to_existing_samples_threshold)[0]
decayed_of_interest = [diclist[x]['decayed'] for x in decayed_of_interest]
decayed_of_interest = np.array(decayed_of_interest)

# %%
if args.verbose:
    print(f'Start looking at closest {args.closest_clusters_count} clusters to the decayed samples of interest')
diclist_close_k = [{'decayed':x,
            'similar_exist_close_k':diclist[decayed_dict[x]]['similar_exist'].copy(),
            'similar_scores_close_k':diclist[decayed_dict[x]]['similar_scores'].copy()} 
            for x in decayed_of_interest]

existing = decayed_array==0
cluster_assignments = []
for i in range(cluster_counts):
    cluster_assignments.append(cluster_assignment == i)

decayed_interest_dict = {}
for i in range(len(decayed_of_interest)):
    decayed_interest_dict[decayed_of_interest[i]] = i

close_k = args.closest_clusters_count
decayed_of_interest_clusters = []
for i in range(len(decayed_of_interest)):
    decayed_ind = decayed_of_interest[i]
    decayed_of_interest_clusters.append(get_top_n_indices(dot_products[decayed_ind],close_k+1)[1:])

clusters_decayed_of_interest = [[] for i in range(cluster_counts)]

for i in range(len(decayed_of_interest)):
    for j in range(close_k):
        clusters_decayed_of_interest[decayed_of_interest_clusters[i][j]].append(decayed_of_interest[i])


clusters_exist = []
for i in range(cluster_counts):
    clusters_exist.append(np.where((cluster_assignment == i) & (decayed_array == 0))[0])

# %%

for i in tqdm(range(cluster_counts)):
    temp = dataset_embeddings[clusters_decayed_of_interest[i]] @ dataset_embeddings[existing & 
                                                        cluster_assignments[i]].T
    for j in range(len(temp)):
        decayed_ind = clusters_decayed_of_interest[i][j]
        similar_inds = get_top_n_indices(temp[j], 10)
        diclist_close_k[decayed_interest_dict[decayed_ind]]['similar_exist_close_k'] = diclist_close_k[decayed_interest_dict[decayed_ind]]['similar_exist_close_k'] + clusters_exist[i][similar_inds].tolist()
        diclist_close_k[decayed_interest_dict[decayed_ind]]['similar_scores_close_k'] = diclist_close_k[decayed_interest_dict[decayed_ind]]['similar_scores_close_k'] + temp[j][similar_inds].tolist()

# %%
for i in tqdm(range(len(decayed_of_interest))):
    similar_scores = np.array(diclist_close_k[i]['similar_scores_close_k'])
    similar_exist = np.array(diclist_close_k[i]['similar_exist_close_k'])
    temp_inds = get_top_n_indices(similar_scores, 10)
    similar_scores = similar_scores[temp_inds]
    similar_exist = similar_exist[temp_inds]
    diclist_close_k[i]['similar_scores_close_k'] = similar_scores.tolist()
    diclist_close_k[i]['similar_exist_close_k'] = similar_exist.tolist()
# %%
good_th = args.similarity_to_existing_samples_threshold
similarities_of_interest_cluster = [diclist_close_k[i]['similar_scores_close_k'][-1] for i in range(len(decayed_of_interest))]
good_ones = np.where(np.array(similarities_of_interest_cluster) < good_th)[0]
good_ones = [diclist_close_k[i]['decayed'] for i in good_ones]
good_ones_len = len(good_ones)
print(f'# decayed indices with similarity less than {good_th}: {good_ones_len}')
# %%
if args.verbose:
    print(f'Apply lower and upper thresholds to the similarity scores of the decayed samples of interest')
good_dataset_embeddings = dataset_embeddings[good_ones]
css = good_dataset_embeddings @ good_dataset_embeddings.T

# We want to ignore the diagonal elements
for i in range(good_ones_len):
    css[i, i] = 0
# We also want to ignore the samples that are too similar to each other
upper_th = args.similarity_to_decayed_samples_upper_threshold + 1e-5
""" for i in range(good_ones_len):
    for j in range(i+1, good_ones_len):
        if css[i, j] > upper_th:
            css[i, j] = 0
            css[j, i] = 0 """
for i in range(good_ones_len):
    for j in range(i+1, good_ones_len):
        if css[i, j] > upper_th:
            css[j, :] = 0
            css[:, j] = 0 
# %%
lower_th = args.similarity_to_decayed_samples_lower_threshold
count_th = args.similar_decayed_samples_count_threshold
new_good_ones = []
for i in range(good_ones_len):
    if np.sum(css[i] > lower_th) > count_th:
        new_good_ones.append(good_ones[i])
relevant_dataset_embeddings = dataset_embeddings[new_good_ones]
relevant_captions = captions[new_good_ones]
relevant_urls = url[new_good_ones]
if args.verbose:
    print(f'# decayed samples fulfill {lower_th} < similarity < {upper_th} and have more than {count_th} similar samples: {len(new_good_ones)}')

# %%
if args.verbose:
    print(f'Save the results')
result_folder = args.result_folder
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

all_captions_file = os.path.join(result_folder, 'all_captions.txt')
with open(all_captions_file, 'w') as fout:
    fout.write('Closest clusters count: ' + str(args.closest_clusters_count) + \
' similarity to existing samples threshold: ' + str(args.similarity_to_existing_samples_threshold) + \
' similarity to decayed samples lower threshold: ' + str(args.similarity_to_decayed_samples_lower_threshold) + \
' similarity to decayed samples upper threshold: ' + str(args.similarity_to_decayed_samples_upper_threshold) + \
' similar decayed samples count threshold: ' + str(args.similar_decayed_samples_count_threshold) + '\n')
    for i in range(len(relevant_captions)):
        fout.write(str(new_good_ones[i]) + ',' + relevant_captions[i] + '\n')
# %%
if args.decayed_sample_clustering:
    n_clusters = args.number_of_decayed_sample_clusters
    kmeans_fitter =  MiniBatchKMeans(n_clusters=n_clusters, batch_size=256 * 2, verbose=0, n_init=5, max_iter=500, random_state=42)
    kmeans = kmeans_fitter.fit(relevant_dataset_embeddings)
    relevant_cluster_centers = kmeans.cluster_centers_
    relevant_dot_products, relevant_distances = dot_products_distances(relevant_dataset_embeddings, relevant_cluster_centers)
    relevant_cluster_assignments = np.argmax(relevant_dot_products, axis=1)
    relevant_cluster_counts = Counter(relevant_cluster_assignments)
    
    """ plt.bar(range(n_clusters), [relevant_cluster_counts[i] for i in range(n_clusters)])
    plt.xlabel("Cluster index")
    plt.ylabel("Percentage of decayed samples")
    plt.title("Percentage of decayed samples in each cluster")
    plt.show() """

    for i in range(n_clusters):
        print(f'# of samples in cluster {i}: {relevant_cluster_counts[i]}')
    
    """ j = 4
    random.seed(42)
    rc = relevant_captions[relevant_cluster_assignments==j]
    ru = relevant_urls[relevant_cluster_assignments==j]
    rands = random.sample(range(len(rc)), min(20, len(rc)))
    for i in rands:
        print(rc[i])
        print(ru[i])
        print() """
    
    new_good_ones2 = np.array(new_good_ones)
    for i in range(n_clusters):
        rids = new_good_ones2[relevant_cluster_assignments==i]
        rc = relevant_captions[relevant_cluster_assignments==i]
    
        cluster_file = os.path.join(result_folder, 'cluster_' + str(i) + '.txt')
        with open(cluster_file, 'w') as fout:
            fout.write('Closest clusters count: ' + str(args.closest_clusters_count) + \
    ' similarity to existing samples threshold: ' + str(args.similarity_to_existing_samples_threshold) + \
    ' similarity to decayed samples lower threshold: ' + str(args.similarity_to_decayed_samples_lower_threshold) + \
    ' similarity to decayed samples upper threshold: ' + str(args.similarity_to_decayed_samples_upper_threshold) + \
    ' similar decayed samples count threshold: ' + str(args.similar_decayed_samples_count_threshold) + '\n')
            for j in range(len(rc)):
                fout.write(str(rids[j]) + ',' + rc[j] + '\n')

# %%
