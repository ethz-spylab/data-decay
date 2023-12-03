# %%
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_new import find_matching_labels_and_clusters

# Add argument class to the args using --dataset_embeddings_path /data/cc3m/cc3m_2023/embeddings/text_embeddings_L14.npy --labels_path /data/cc3m/imagenet_classes.txt --label_embeddings_path /data/cc3m/cc3m_2023/embeddings/imagenet_class_embeddings_L14.npy --decayed_indices_path /data/cc3m/decayed_indices.txt --clusters_folder /data/cc3m/script_tests/clusters/

class Args():
    def __init__(self):
        self.dataset_embeddings_path = '/data/cc3m/cc3m_2023/embeddings/text_embeddings_L14.npy'
        self.labels_path = '/data/cc3m/imagenet_classes.txt'
        self.label_embeddings_path = '/data/cc3m/cc3m_2023/embeddings/imagenet_class_embeddings_L14.npy'
        self.decayed_indices_path = '/data/cc3m/decayed_indices.txt'
        self.clusters_folder = '/data/cc3m/script_tests/clusters/'
        self.similarity_type = 'dot_products'
        self.sample_count_threshold = 1000
        self.label_percentage_in_cluster_threshold = 0.5
        self.cluster_percentage_in_label_threshold = 0.04
        self.decayed_percentage_threshold = 0.0
        self.plot = True
        self.summary = True
        self.verbose = 1

args = Args()

""" parser = argparse.ArgumentParser(description='Find matching labels and clusters that decayed together. This assumes that the clusters have already been found and similarities/distances are calculated.\
                                 Each sample is assigned to the cluster with the highest similarity or smallest distance. \
                                 We require label embeddings to be calculated beforehand. For the sample embeddings, we use either the caption or image embeddings.')

parser.add_argument('--dataset_embeddings_path', type=str, default='embeddings/text_embeddings.npy',
                    help='embeddings path for the dataset. This can be either the image or caption embeddings.')
parser.add_argument('--labels_path', type=str, default='labels.txt',
                    help='labels path')
parser.add_argument('--label_embeddings_path', type=str, default='embeddings/labels_embeddings.npy',
                    help='label embeddings path')
parser.add_argument('--clusters_folder', type=str, default='clusters/',
                    help='clusters folder')
parser.add_argument('--similarity_type', type=str, default='dot_products',
                    help='similarity type. Either dot_products or distances. If distances is chosen, 1 - distances is used.')
parser.add_argument('--decayed_indices_path', type=str, default='decayed_indices.txt',
                    help='decayed indices path')
parser.add_argument('--sample_count_threshold', type=int, default=1000,
                    help='how many times a label should be present in the dataset to be considered')
parser.add_argument('--label_percentage_in_cluster_threshold', type=float, default=0.5,
                    help='threshold percentage of label elements in a cluster to be considered as matching \
                    (e.g. %70 of "car" labeled elements in cluster 1)')
parser.add_argument('--cluster_percentage_in_label_threshold', type=float, default=0.04,
                    help='threshold for what percentage of a cluster should consist of a label to be considered as matching \
                    (e.g. %70 of cluster 1 should consist of "car" labeled elements)')
parser.add_argument('--decayed_percentage_threshold', type=float, default=0.0,
                    help='threshold percentage of decayed indices in a cluster to be considered as matching')
parser.add_argument('--plot', type=bool, default=True,
                    help='whether to plot the results or not')
parser.add_argument('--summary', type=bool, default=True,
                    help='whether to print the summary or not')
parser.add_argument('--verbose', type=int, default=1,
                    help='verbosity of code')

args = parser.parse_args() """


# %%
# Read the labels
labels_path = args.labels_path
labels = []
with open(labels_path, "r") as f:
    for line in f:
        labels.append(line.strip())

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

cluster_count = cluster_centers.shape[0]
if args.verbose:
    print("Number of clusters: {}".format(cluster_count))

# Load the list of decayed indices
decayed_indices_path = args.decayed_indices_path
decayed_indices = []
with open(decayed_indices_path, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))

# Load the dataset embeddings
if args.verbose:
    print("Loading dataset embeddings")
dataset_embeddings_path = args.dataset_embeddings_path
dataset_embeddings = np.load(dataset_embeddings_path)

# Load the label embeddings
if args.verbose:
    print("Loading label embeddings")
label_embeddings_path = args.label_embeddings_path
label_embeddings = np.load(label_embeddings_path)

# Find dataset and label embeddings comparison
assert dataset_embeddings.shape[1] == label_embeddings.shape[1], "Dataset and label embeddings should have the same dimension."
dataset_vs_label = np.matmul(dataset_embeddings, label_embeddings.T)

# Find the label assignments using argmax
label_assignment = np.argmax(dataset_vs_label, axis=1)

# %%
# Find the matching labels and clusters
if args.verbose:
    print("Finding matching labels and clusters")
_, _ = find_matching_labels_and_clusters(cluster_assignment,
                                         label_assignment, decayed_indices,
                                         labels, labels,
                                         label_element_count_threshold  = args.sample_count_threshold,
                                         label_percentage_in_cluster_threshold  = args.label_percentage_in_cluster_threshold,
                                         cluster_percentage_in_label_threshold  = args.cluster_percentage_in_label_threshold,
                                         decay_percentage_of_label_in_cluster_threshold = args.decayed_percentage_threshold,
                                         plot = args.plot,
                                         summary = args.summary)
# %%

# kodu terminalden calistirirken bar hata veriyor, onu coz. ek olarak plotlari kaydetmek icin bir yol bul