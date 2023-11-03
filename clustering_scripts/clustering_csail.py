#%%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
CSAIL_PLACES = DATA_FOLDER / "csail_places.txt"
CSAIL_PLACES_EMBEDDINGS = EMBEDDINGS_FOLDER / "csail_places_embeddings.npy"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
CC_VS_CSAIL_PLACES = EMBEDDINGS_FOLDER / "CC_vs_csail_places.npy"
CC_VS_CSAIL_PLACES_ASSIGNMENT = EMBEDDINGS_FOLDER / "CC_vs_csail_places_assignment.npy"

DECAYED_INDICES = DATA_FOLDER / "decayed_indices.txt"
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (plot_missing_num_perc, get_relevant_captions_and_urls, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters,get_label_cluster_matching_captions_urls)
import pickle
import torch
CLUSTER_CENTERS = EMBEDDINGS_FOLDER / "cluster_centers_L14.npy"
DOT_PRODUCTS = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_L14.npy"
DISTANCES = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_distances_L14.npy"

CSAIL_PLACES_COUNT = 476

from tqdm import tqdm
from urllib.parse import urlparse
from collections import Counter

IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"

from transformers import CLIPProcessor, CLIPModel
# %%

csail_places = []
with open(CSAIL_PLACES, 'r') as f:
    for line in f:
        csail_places.append("This is a " + line.strip())


# %%

cluster_centers = np.load(CLUSTER_CENTERS)
dot_products = np.load(DOT_PRODUCTS)
csail_places_embeddings = np.load(CSAIL_PLACES_EMBEDDINGS)
# %%
cluster_assignment = np.argmax(dot_products, axis=1)
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
captions = np.array(cc_captions["caption"])
url = np.array(cc_captions["url"])
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

get_relevant_captions_and_urls(dot_products, 30, only_argmax=True, sort_best=False)
# %%
csail_places_assignment = np.load(CC_VS_CSAIL_PLACES_ASSIGNMENT)
# %%
# TODO: find how many elements are there in each csail place
csail_places_element_counts = np.zeros(CSAIL_PLACES_COUNT)
for i in range(CSAIL_PLACES_COUNT):
    csail_places_element_counts[i] = np.count_nonzero(csail_places_assignment == i)
# %%
# TODO: print the number of csail places with 0 elements
print(np.count_nonzero(csail_places_element_counts == 0))

# %%
# TODO: find percentage of decayed indices in each csail place
decayed_in_csail_places = np.zeros(CSAIL_PLACES_COUNT, dtype=int)
for i in range(CSAIL_PLACES_COUNT):
    decayed_in_csail_places[i] = np.count_nonzero(csail_places_assignment[decayed_indices] == i)
percentage_of_decayed_indices_in_csail_places = decayed_in_csail_places / csail_places_element_counts

# %%
highest_percentage_csail_places = get_top_n_indices(percentage_of_decayed_indices_in_csail_places,10)
# %%
fig = plot_missing_num_perc(highest_percentage_csail_places,
                            decayed_in_csail_places,
                            percentage_of_decayed_indices_in_csail_places,
                            title='CSAIL places with highest percentage of missing')
plt.show()

# %%
csail_places_short = []
with open(CSAIL_PLACES, 'r') as f:
    for line in f:
        csail_places_short.append(line.strip())

# %%
relevant_labels, relevant_clusters = find_matching_labels_and_clusters(cluster_assignment,
                                         csail_places_assignment, decayed_indices,
                                         csail_places_short, csail_places_short,
                                         imagenet_element_count_threshold  = 1000,
                                         imagenet_percentage_in_cluster_threshold  = 0.3,
                                         cluster_percentage_in_imagenet_threshold  = 0.3,
                                         decay_percentage_of_label_in_cluster_threshold = 0.15)
# %%
get_relevant_captions_and_urls(dot_products, 30, only_argmax=True, sort_best=False)

# %%

csail_places_embeddings = np.load(CSAIL_PLACES_EMBEDDINGS)

# %%

# For cluster 30 / 100
relevant_cluster = 30
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:0"
model.to(device)
text = ["This is a guitar","This is a musical instrument",
        "This is a stage","This is a concert",
        "This is a indoor","This is a electric guitar",
        "This is a guitar on stage"]
inputs = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
with torch.no_grad():
    outputs = model.text_model(**inputs)
    txt_embeds = outputs[1]
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
txt_embeds = txt_embeds.cpu().numpy().astype('float32')

txt_vs_cluster = txt_embeds @ cluster_centers[relevant_cluster].T

#Print each text and its similarity to the cluster center
for i in range(len(text)):
    print(f'Text: {text[i]}, \
            \n similarity: {txt_vs_cluster[i]:.3f}')


sargmax = np.argmax(txt_vs_cluster)
print(f'Most similar description to cluster center: {text[sargmax]}, \
      \n similarity: {txt_vs_cluster[sargmax]:.3f}')

label_vs_cluster = csail_places_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {csail_places[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster):.3f}')
# %%
