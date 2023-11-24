#%%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
IMAGENET_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "imagenet_class_embeddings_L14.npy"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
# CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings.npy"
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
CC_VS_IMAGENET = EMBEDDINGS_FOLDER / "CC_vs_imagenet_L14.npy"
DECAYED_INDICES = DATA_FOLDER / "decayed_indices.txt"
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import (plot_missing_num_perc, get_relevant_captions, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters)
import pickle
import torch
CLUSTER_CENTERS = EMBEDDINGS_FOLDER / "cluster_centers_L14.npy"
DOT_PRODUCTS = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_L14.npy"
DISTANCES = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_distances_L14.npy"

IMAGENET_LABEL_COUNT = 1000

from tqdm import tqdm

IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"


# %%
#TODO: read IMAGENET_CLASSES_SHORT and save it to a list
imagenet_classes_short = []
with open(IMAGENET_CLASSES_SHORT, "r") as f:
    for line in f:
        imagenet_classes_short.append(line.strip())

imagenet_classes_long = []
with open(IMAGENET_CLASSES_LONG, "r") as f:
    for line in f:
        imagenet_classes_long.append(line.strip())


# %%
cluster_centers = np.load(CLUSTER_CENTERS)
dot_products = np.load(DOT_PRODUCTS)
distances = np.load(DISTANCES)


# %%
# TODO: check if dot products make sense
get_relevant_captions(dot_products, 29, only_argmax=True, sort_best=False)

# %%
cluster_assignment = np.argmax(dot_products, axis=1)

# %%
number_of_clusters = dot_products.shape[1]
# %%
# TODO: find how many elements are there in each cluster
cluster_element_counts = np.zeros(number_of_clusters)
for i in range(number_of_clusters):
    cluster_element_counts[i] = np.count_nonzero(cluster_assignment == i)

# %%
#TODO: plot the distribution of the clusters

plt.hist(cluster_element_counts, bins=number_of_clusters)
plt.title('Distribution of the clusters')
plt.show()

# %%
# TODO: plot the 10 clusters with the highest number of elements in log scale
highest_number_clusters = get_top_n_indices(cluster_element_counts,10)
plt.bar([str(x) for x in highest_number_clusters], cluster_element_counts[highest_number_clusters])
plt.yscale('log')
plt.title('10 clusters with highest number of elements')
plt.show()


# %% 
# TODO: read decayed indices to a list
decayed_indices = []
with open(DECAYED_INDICES, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))
print("Number of decayed indices: ", len(decayed_indices))

# %%
cc_captions = pd.read_csv(CC_CAPTIONS_DF, sep="\t", header=None)
cc_captions.columns = ["caption", "url"]
captions = cc_captions["caption"].tolist()
url = cc_captions["url"].tolist()

# %%
# TODO: list first 10 urls of decayed indices
print("First 10 urls of decayed indices:")
for i in range(10):
    print(url[decayed_indices[i]])

# %%
# TODO: find percentage of decayed indices in each cluster
decayed_in_clusters = np.zeros(number_of_clusters, dtype=int)
for i in range(number_of_clusters):
    decayed_in_clusters[i] = np.count_nonzero(cluster_assignment[decayed_indices] == i)
percentage_of_decayed_indices_in_clusters = decayed_in_clusters / cluster_element_counts

# %%
# TODO: bar plot the percentage of decayed indices for each cluster
plt.bar(np.arange(number_of_clusters), percentage_of_decayed_indices_in_clusters)
plt.title('Percentage of decayed indices in each cluster')
plt.show()

# %%
# TODO: find the 10 clusters with the highest percentage of decayed indices and print them and their percentages
highest_percentage_clusters = get_top_n_indices(percentage_of_decayed_indices_in_clusters,10)
print(highest_percentage_clusters)
print(percentage_of_decayed_indices_in_clusters[highest_percentage_clusters])


# %%

fig = plot_missing_num_perc(highest_percentage_clusters,
                            decayed_in_clusters, 
                            percentage_of_decayed_indices_in_clusters,
                            title='Clusters with highest percentage of missing')
plt.show()

# %%
# TODO: find the 10 clusters with the highest number of decayed indices
highest_number_decayed = get_top_n_indices(decayed_in_clusters)

fig = plot_missing_num_perc(highest_number_decayed, 
                   decayed_in_clusters, 
                   percentage_of_decayed_indices_in_clusters, 
                   'Clusters with highest number of missing')
plt.show()


# %%
cc_vs_imagenet = np.load(CC_VS_IMAGENET)
# %%
cc_vs_imagenet.shape
# %%
imagenet_assignment = np.argmax(cc_vs_imagenet, axis=1)
imagenet_assignment.shape


# %%
# TODO: find how many elements are there in each imagenet assignment
imagenet_element_counts = np.zeros(IMAGENET_LABEL_COUNT)
for i in range(IMAGENET_LABEL_COUNT):
    imagenet_element_counts[i] = np.count_nonzero(imagenet_assignment == i)


# TODO: find the number of imagenet_element_counts with 0 elements
print("Number of imagenet_element_counts with 0 elements: ", np.count_nonzero(imagenet_element_counts == 0))


#%%

#plot_cluster_make_up(30, cluster_assignment, imagenet_assignment, decayed_indices, order="number")
#plot_cluster_make_up(30, cluster_assignment, imagenet_assignment, decayed_indices, order="percentage")

# %%

#plot_cluster_make_up(29, cluster_assignment, imagenet_assignment, decayed_indices, order="number")
#plot_cluster_make_up(29, cluster_assignment, imagenet_assignment, decayed_indices, order="percentage")

# %%
# TODO: find cluster to caption assignment and print the caption for each cluster

cluster_to_caption_assignment = np.argmax(dot_products, axis=0)
for i in range(number_of_clusters):
    print("Cluster: ", i, " Match: ", 
          dot_products[cluster_to_caption_assignment[i],i],
          " # elements", int(cluster_element_counts[i]),
          " Caption: ", captions[cluster_to_caption_assignment[i]])
    
# %%

relevant_labels, relevant_clusters = find_matching_labels_and_clusters(cluster_assignment,
                                         imagenet_assignment, decayed_indices,
                                         imagenet_classes_short, imagenet_classes_long,
                                         imagenet_element_count_threshold  = 1000,
                                         imagenet_percentage_in_cluster_threshold  = 0.5,
                                         cluster_percentage_in_imagenet_threshold  = 0.4,)


