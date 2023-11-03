#%%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
IMAGENET_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "imagenet_class_embeddings_L14.npy"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
# CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings.npy"
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
CC_VS_IMAGENET = EMBEDDINGS_FOLDER / "CC_vs_imagenet_L14.npy"
CC_VS_IMAGENET_FAST = EMBEDDINGS_FOLDER / "CC_vs_imagenet_L14_fast.npy"
CC_VS_IMAGENET_ASSIGNMENT = EMBEDDINGS_FOLDER / "CC_vs_imagenet_assignment_L14.npy"
DECAYED_INDICES = DATA_FOLDER / "decayed_indices.txt"
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (plot_missing_num_perc, get_relevant_captions_and_urls, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters,get_label_cluster_matching_captions_urls,
    fast_load, fast_save, dot_products_distances)
import pickle
import torch
CLUSTER_CENTERS = EMBEDDINGS_FOLDER / "cluster_centers_L14.npy"
DOT_PRODUCTS = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_L14.npy"
DISTANCES = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_distances_L14.npy"

IMAGENET_LABEL_COUNT = 1000
CSAIL_PLACES_COUNT = 476
IMAGENET21K_LABEL_COUNT = 21843
STEP_SIZE = 1000

from tqdm import tqdm
from urllib.parse import urlparse
from collections import Counter
import random

from sklearn.cluster import KMeans, MiniBatchKMeans
import os
import json

from transformers import CLIPProcessor, CLIPModel

IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"


CC_VS_CSAIL_PLACES = EMBEDDINGS_FOLDER / "CC_vs_csail_places.npy"
CC_VS_CSAIL_PLACES_FAST = EMBEDDINGS_FOLDER / "CC_vs_csail_places_fast.npy"
CC_VS_CSAIL_PLACES_ASSIGNMENT = EMBEDDINGS_FOLDER / "CC_vs_csail_places_assignment.npy"
CSAIL_PLACES = DATA_FOLDER / "csail_places.txt"

IMAGENET_EMBEDDINGS_FOLDER = CC_VS_CSAIL_PLACES
IMAGENET_CLASSES_SHORT = CSAIL_PLACES
IMAGENET_CLASSES_LONG = CSAIL_PLACES
IMAGENET_LABEL_COUNT = CSAIL_PLACES_COUNT
CC_VS_IMAGENET = CC_VS_CSAIL_PLACES
CC_VS_IMAGENET_FAST = CC_VS_CSAIL_PLACES_FAST
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
#distances = np.load(DISTANCES)
imagenet_label_embeddings = np.load(IMAGENET_EMBEDDINGS_FOLDER)
cc_vs_imagenet = fast_load(str(CC_VS_IMAGENET))
cc_embeddings = fast_load(str(CC_EMBEDDINGS_FOLDER))
# %%
print(f'cc_vs_imagenet shape: {cc_vs_imagenet.shape}')
print(f'cc_embeddings shape: {cc_embeddings.shape}')
# %%
# TODO: read decayed indices to a list
decayed_indices = []
with open(DECAYED_INDICES, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))
print("Number of decayed indices: ", len(decayed_indices))
decayed_array = np.zeros(cc_embeddings.shape[0])
decayed_array[decayed_indices] = 1
# %%
cc_captions = pd.read_csv(CC_CAPTIONS_DF, sep="\t", header=None)
cc_captions.columns = ["caption", "url"]
captions = np.array(cc_captions["caption"])
url = np.array(cc_captions["url"])
# %%
threshold = 0.5
imagenet_to_cc = []
for i in tqdm(range(IMAGENET_LABEL_COUNT)):
    imagenet_to_cc.append(np.where(cc_vs_imagenet[:, i] > threshold)[0])
cc_to_imagenet = []
for i in tqdm(range(cc_vs_imagenet.shape[0])):
    cc_to_imagenet.append(np.where(cc_vs_imagenet[i, :] > threshold)[0])
# %%
decayed_cc_to_imagenet = [cc_to_imagenet[i] for i in decayed_indices]
# %%
decayed_imagenet_to_cc = []
decayed_cc_vs_imagenet = [cc_vs_imagenet[x] for x in decayed_indices]
decayed_cc_vs_imagenet = np.array(decayed_cc_vs_imagenet)
for i in tqdm(range(IMAGENET_LABEL_COUNT)):
    decayed_imagenet_to_cc.append([decayed_indices[x] for x in np.where(decayed_cc_vs_imagenet[:, i] > threshold)[0]])
# %%
cc_distribution_to_imagenet = np.zeros(IMAGENET_LABEL_COUNT)
decayed_cc_distribution_to_imagenet = np.zeros(IMAGENET_LABEL_COUNT)
for i in tqdm(range(cc_vs_imagenet.shape[0])):
    cc_distribution_to_imagenet[cc_to_imagenet[i]] += 1
for i in tqdm(range(len(decayed_indices))):
    decayed_cc_distribution_to_imagenet[decayed_cc_to_imagenet[i]] += 1
# %%
percentage_decayed_in_imagenet = decayed_cc_distribution_to_imagenet / cc_distribution_to_imagenet
percentage_decayed_in_imagenet[np.isnan(percentage_decayed_in_imagenet)] = 0


# %%
k = 20
num_threshold = 10
percentage_decayed_in_imagenet_cp = percentage_decayed_in_imagenet.copy()
percentage_decayed_in_imagenet_cp[cc_distribution_to_imagenet < num_threshold] = 0
highest_percentage_num_threshold = get_top_n_indices(percentage_decayed_in_imagenet_cp, k)
# %%
fig = plot_missing_num_perc(highest_percentage_num_threshold, 
                      cc_distribution_to_imagenet,
                      percentage_decayed_in_imagenet,
                      labels = [imagenet_classes_short[x] for x in highest_percentage_num_threshold])

plt.show()
for i in highest_percentage_num_threshold:
    print(f'# of samples: {int(cc_distribution_to_imagenet[i])}, \
    % of decayed: {percentage_decayed_in_imagenet[i]*100:.2f}, \
    name: {imagenet_classes_long[i]}')
# %%
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)


# %%
text = ["This is a room advert", "This is a guest room advert",
        "This is a air conditioner"]
inputs = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
with torch.no_grad():
    outputs = model.text_model(**inputs)
    txt_embeds = outputs[1]
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
txt_embeds = txt_embeds.cpu().numpy().astype('float32')


txt_vs_embeddings = txt_embeds @ cc_embeddings.T

decayed_txt_vs_embeddings = txt_vs_embeddings[:,decayed_array==1]
existing_txt_vs_embeddings = txt_vs_embeddings[:,decayed_array==0]

# %%
for i in range(len(text)):

    txt = i
    th = 0.5
    plt.hist(decayed_txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.hist(existing_txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.legend(["Decayed", "Existing"])
    plt.xlabel("Similarity score")
    plt.ylabel("Count")
    plt.title("\""+text[txt]+"\""+" vs captions")
    plt.show()

    dec = np.sum(decayed_txt_vs_embeddings[txt] > th)
    ex = np.sum(existing_txt_vs_embeddings[txt] > th)
    print(i)
    print(f'# of decayed: {dec}, \
        \n# of total: {ex+dec}, \
        \n% of decayed: {dec/(ex+dec)*100:.2f}')
# %%

# %%

txt_vs_embeddings = cc_vs_imagenet[:, highest_percentage_num_threshold].T

decayed_txt_vs_embeddings = txt_vs_embeddings[:,decayed_array==1]
existing_txt_vs_embeddings = txt_vs_embeddings[:,decayed_array==0]

# %%
for i in range(len(highest_percentage_num_threshold)):

    txt = i
    th = 0.5
    plt.hist(decayed_txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.hist(existing_txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.legend(["Decayed", "Existing"])
    plt.xlabel("Similarity score")
    plt.ylabel("Count")
    plt.title("\""+imagenet_classes_long[highest_percentage_num_threshold[txt]]+"\""+" vs captions")
    plt.show()

    dec = np.sum(decayed_txt_vs_embeddings[txt] > th)
    ex = np.sum(existing_txt_vs_embeddings[txt] > th)
    print(i)
    print(f'# of decayed: {dec}, \
        \n# of total: {ex+dec}, \
        \n% of decayed: {dec/(ex+dec)*100:.2f}')
# %%






# %%
# txt_vs_embeddings e bakmak icin
k = 0
chosen = np.where(txt_vs_embeddings[k] > 0.5)[0]
relevant_cc_embeddings = cc_embeddings[chosen]
relevant_captions = captions[chosen]
relevant_urls = url[chosen]
relevant_decay_positions = np.where(decayed_array[chosen]==1)
# %%
N_CLUSTERS = 10
kmeans_fitter =  MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=256 * 2, verbose=0, n_init=5, max_iter=500, random_state=42)
kmeans = kmeans_fitter.fit(relevant_cc_embeddings)
relevant_cluster_centers = kmeans.cluster_centers_
relevant_dot_products, relevant_distances = dot_products_distances(relevant_cc_embeddings, relevant_cluster_centers)
relevant_cluster_assignments = np.argmax(relevant_dot_products, axis=1)
decayed_relent_cluster_assignments = relevant_cluster_assignments[relevant_decay_positions]
relevant_cluster_counts = Counter(relevant_cluster_assignments)
decayed_relevant_cluster_counts = Counter(decayed_relent_cluster_assignments)
percentage_decayed_relevant_cluster_counts = np.array([decayed_relevant_cluster_counts[x] / relevant_cluster_counts[x] for x in range(N_CLUSTERS)])
# %%
plt.bar(range(N_CLUSTERS), percentage_decayed_relevant_cluster_counts)
plt.xlabel("Cluster index")
plt.ylabel("Percentage of decayed samples")
plt.title("Percentage of decayed samples in each cluster")
plt.show()

for i in range(N_CLUSTERS):
    print(f'# of samples in cluster {i}: {relevant_cluster_counts[i]}, \
    % of decayed: {percentage_decayed_relevant_cluster_counts[i]*100:.2f}')
# %%
j = 6
random.seed(42)
rc = relevant_captions[relevant_cluster_assignments==j]
ru = relevant_urls[relevant_cluster_assignments==j]
rands = random.sample(range(len(rc)), 50)
for i in rands:
    if decayed_array[chosen][relevant_cluster_assignments==j][i] == 1:
        print("DECAYED")
    else:
        print("EXISTING")
    print(rc[i])
    print(ru[i])
    print()













# %%

random.seed(42)
k = 6
selection = highest_percentage_num_threshold.copy()
relevant_captions = captions[imagenet_to_cc[selection[k]]]
relevant_urls = url[imagenet_to_cc[selection[k]]]
rands = random.sample(range(len(relevant_captions)), 10)
for i in rands:
    if decayed_array[imagenet_to_cc[selection[k]][i]] == 1:
        print("DECAYED")
    else:
        print("EXISTS")
    print(relevant_captions[i])
    print(relevant_urls[i])
    print()

# %%
k = 13
#selection = highest_percentage_num_threshold.copy()
selection = highest_percentage_num_threshold.copy()
relevant_cc_embeddings = cc_embeddings[imagenet_to_cc[selection[k]]]
relevant_captions = captions[imagenet_to_cc[selection[k]]]
relevant_urls = url[imagenet_to_cc[selection[k]]]
relevant_decay_positions = [np.where(imagenet_to_cc[selection[k]] == x)[0][0] for x in decayed_imagenet_to_cc[selection[k]]]
# %%
N_CLUSTERS = 20
kmeans_fitter =  MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=256 * 2, verbose=0, n_init=5, max_iter=500, random_state=42)
kmeans = kmeans_fitter.fit(relevant_cc_embeddings)
relevant_cluster_centers = kmeans.cluster_centers_
relevant_dot_products, relevant_distances = dot_products_distances(relevant_cc_embeddings, relevant_cluster_centers)
relevant_cluster_assignments = np.argmax(relevant_dot_products, axis=1)
decayed_relent_cluster_assignments = relevant_cluster_assignments[relevant_decay_positions]
relevant_cluster_counts = Counter(relevant_cluster_assignments)
decayed_relevant_cluster_counts = Counter(decayed_relent_cluster_assignments)
percentage_decayed_relevant_cluster_counts = np.array([decayed_relevant_cluster_counts[x] / relevant_cluster_counts[x] for x in range(N_CLUSTERS)])
# %%
plt.bar(range(N_CLUSTERS), percentage_decayed_relevant_cluster_counts)
plt.xlabel("Cluster index")
plt.ylabel("Percentage of decayed samples")
plt.title("Percentage of decayed samples in each cluster")
plt.show()

for i in range(N_CLUSTERS):
    print(f'# of samples in cluster {i}: {relevant_cluster_counts[i]}, \
    % of decayed: {percentage_decayed_relevant_cluster_counts[i]*100:.2f}')

# %%
j = 16
random.seed(42)
rc = relevant_captions[relevant_cluster_assignments==j]
ru = relevant_urls[relevant_cluster_assignments==j]
rands = random.sample(range(len(rc)), min(50, len(rc)))
for i in rands:
    if decayed_array[imagenet_to_cc[selection[k]][relevant_cluster_assignments==j][i]] == 1:
        print("DECAYED")
    else:
        print("EXISTS")
    print(rc[i])
    print(ru[i])
    print()
# %%
