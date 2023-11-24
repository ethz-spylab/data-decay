#%%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
CC_EMBEDDINGS_FOLDER_FAST = EMBEDDINGS_FOLDER / "text_embeddings_L14_fast.npy"
#CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings.npy"
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
DECAYED_INDICES = DATA_FOLDER / "decayed_indices.txt"
CC_DECAYED_SIMILARITY_DICT = DATA_FOLDER / "cc_decayed_similarity_dict.json"
CC_DECAYED_SIMILARITY_DICT_100 = DATA_FOLDER / "cc_decayed_similarity_dict_100.json" #100 clusters

CLUSTER_CENTERS = EMBEDDINGS_FOLDER / "cluster_centers_L14.npy"
DOT_PRODUCTS = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_L14.npy"
DISTANCES = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_distances_L14.npy"

from transformers import CLIPProcessor, CLIPModel

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import random
from collections import Counter

from tqdm import tqdm
import json

CC_VS_IMAGENET_ASSIGNMENT = EMBEDDINGS_FOLDER / "CC_vs_imagenet_assignment_L14.npy"
IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"

CC_VS_CSAIL_PLACES = EMBEDDINGS_FOLDER / "CC_vs_csail_places.npy"

CC_VS_IMAGENET = EMBEDDINGS_FOLDER / "CC_vs_imagenet_L14.npy"
CC_VS_IMAGENET_FAST = EMBEDDINGS_FOLDER / "CC_vs_imagenet_L14_fast.npy"

from utils import (plot_missing_num_perc, get_relevant_captions_and_urls, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters,get_label_cluster_matching_captions_urls,
    fast_load, fast_save, dot_products_distances)
# %%
cc_captions = pd.read_csv(CC_CAPTIONS_DF, sep="\t", header=None)
cc_captions.columns = ["caption", "url"]
captions = np.array(cc_captions["caption"])
url = np.array(cc_captions["url"])
# %%
cc_embeddings = fast_load(str(CC_EMBEDDINGS_FOLDER_FAST))
cluster_centers = np.load(CLUSTER_CENTERS)
dot_products = np.load(DOT_PRODUCTS)
cluster_counts = dot_products.shape[1]

cc_vs_imagenet = fast_load(str(CC_VS_IMAGENET_FAST))

# %%
cluster_assignment = np.argmax(dot_products, axis=1)
imagenet_assignment = np.load(CC_VS_IMAGENET_ASSIGNMENT)
imagenet_classes_short = []
with open(IMAGENET_CLASSES_SHORT, "r") as f:
    for line in f:
        imagenet_classes_short.append(line.strip())

imagenet_classes_long = []
with open(IMAGENET_CLASSES_LONG, "r") as f:
    for line in f:
        imagenet_classes_long.append(line.strip())

# %% 
# TODO: read decayed indices to a list
decayed_indices = []
with open(DECAYED_INDICES, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))

decayed_array = np.zeros(len(cluster_assignment))
decayed_array[decayed_indices] = 1
# %%
"""diclist holds the decayed indice, indices of 10 most similar existing captions 
in the same cluster as the decayed indice, and the similarity scores of those captions."""
with open(CC_DECAYED_SIMILARITY_DICT_100, 'r') as fin:
    diclist = json.load(fin)
# %%
decayed_dict = {}
for i in range(len(decayed_indices)):
    decayed_dict[decayed_indices[i]] = i
# %%
similarities = [diclist[i]['similar_scores'][-1] for i in range(len(diclist))]
plt.hist(similarities, bins=100)
plt.xlabel("Similarity score of 10th most similar in the same cluster")
plt.ylabel("Count")
plt.show()
# %%
k = np.linspace(0, 1, 100)
dist = [np.sum(np.array(similarities) < i)/len(similarities) for i in k]
plt.plot(k, dist)
plt.xlabel("Similarity score of 10th most similar in the same cluster")
plt.ylabel("Ratio")
plt.title("Distribution of similarity scores")
plt.show()
# %%
threshold = 0.5
print(f'% decayed indices with similarity less than {threshold}: {np.sum(np.array(similarities) < threshold)/len(similarities)*100:.3f}')
print(f'# decayed indices with similarity less than {threshold}: {np.sum(np.array(similarities) < threshold)}')
# %%

threshold = 0.50
print(f'% decayed indices with similarity less than {threshold}: {np.sum(np.array(similarities) < threshold)/len(similarities)*100:.3f}')
print(f'# decayed indices with similarity less than {threshold}: {np.sum(np.array(similarities) < threshold)}')

# %%
decayed_of_interest = np.where(np.array(similarities) < threshold)[0]
decayed_of_interest = [diclist[x]['decayed'] for x in decayed_of_interest]
decayed_of_interest = np.array(decayed_of_interest)

# %%
diclist_close_3 = [{'decayed':x,
            'similar_exist_close_3':diclist[decayed_dict[x]]['similar_exist'].copy(),
            'similar_scores_close_3':diclist[decayed_dict[x]]['similar_scores'].copy()} 
            for x in decayed_of_interest]

existing = decayed_array==0
cluster_assignments = []
for i in range(cluster_counts):
    cluster_assignments.append(cluster_assignment == i)

decayed_interest_dict = {}
for i in range(len(decayed_of_interest)):
    decayed_interest_dict[decayed_of_interest[i]] = i

close_k = 3
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
    temp = cc_embeddings[clusters_decayed_of_interest[i]] @ cc_embeddings[existing & 
                                                        cluster_assignments[i]].T
    for j in range(len(temp)):
        decayed_ind = clusters_decayed_of_interest[i][j]
        similar_inds = get_top_n_indices(temp[j], 10)
        diclist_close_3[decayed_interest_dict[decayed_ind]]['similar_exist_close_3'] = diclist_close_3[decayed_interest_dict[decayed_ind]]['similar_exist_close_3'] + clusters_exist[i][similar_inds].tolist()
        diclist_close_3[decayed_interest_dict[decayed_ind]]['similar_scores_close_3'] = diclist_close_3[decayed_interest_dict[decayed_ind]]['similar_scores_close_3'] + temp[j][similar_inds].tolist()

# %%
for i in tqdm(range(len(decayed_of_interest))):
    similar_scores = np.array(diclist_close_3[i]['similar_scores_close_3'])
    similar_exist = np.array(diclist_close_3[i]['similar_exist_close_3'])
    temp_inds = get_top_n_indices(similar_scores, 10)
    similar_scores = similar_scores[temp_inds]
    similar_exist = similar_exist[temp_inds]
    diclist_close_3[i]['similar_scores_close_3'] = similar_scores.tolist()
    diclist_close_3[i]['similar_exist_close_3'] = similar_exist.tolist()

# %%
""" k = np.linspace(0, 1, 101)
similarities_of_interest_cluster = [diclist[decayed_dict[decayed_of_interest[i]]]['similar_scores'][-1] for i in range(len(decayed_of_interest))]
dist = [np.sum(np.array(similarities_of_interest_cluster) < i) for i in k]
plt.plot(k, dist)
plt.xlabel("Similarity score of 10th most similar in the same cluster")
plt.ylabel("#")
plt.title("Distribution of similarity scores")
plt.show() """
# %%
""" k = np.linspace(0, 1, 101)
similarities_of_interest_cluster = [diclist_close_3[i]['similar_scores_close_3'][-1] for i in range(len(decayed_of_interest))]
dist = [np.sum(np.array(similarities_of_interest_cluster) < i) for i in k]
plt.plot(k, dist)
plt.xlabel("Similarity score of 10th most similar in the same cluster + 3 closest")
plt.ylabel("#")
plt.title("Distribution of similarity scores")
plt.show() """









# %%
good_th = 0.6
similarities_of_interest_cluster = [diclist_close_3[i]['similar_scores_close_3'][-1] for i in range(len(decayed_of_interest))]
good_ones = np.where(np.array(similarities_of_interest_cluster) < good_th)[0]
good_ones = [diclist_close_3[i]['decayed'] for i in good_ones]
print(len(good_ones))
# %%
good_cc_embeddings = cc_embeddings[good_ones]
css = good_cc_embeddings @ good_cc_embeddings.T
# %%
for i in range(len(good_ones)):
    css[i, i] = 0

# %%
new_th = 0.8
new_good_ones = []
for i in range(len(good_ones)):
    if np.sum(css[i] > new_th) > 10:
        new_good_ones.append(good_ones[i])
relevant_cc_embeddings = cc_embeddings[new_good_ones]
relevant_captions = captions[new_good_ones]
relevant_urls = url[new_good_ones]
print(len(new_good_ones))
# %%
N_CLUSTERS = 10
kmeans_fitter =  MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=256 * 2, verbose=0, n_init=5, max_iter=500, random_state=42)
kmeans = kmeans_fitter.fit(relevant_cc_embeddings)
relevant_cluster_centers = kmeans.cluster_centers_
relevant_dot_products, relevant_distances = dot_products_distances(relevant_cc_embeddings, relevant_cluster_centers)
relevant_cluster_assignments = np.argmax(relevant_dot_products, axis=1)
relevant_cluster_counts = Counter(relevant_cluster_assignments)
# %%
plt.bar(range(N_CLUSTERS), [relevant_cluster_counts[i] for i in range(N_CLUSTERS)])
plt.xlabel("Cluster index")
plt.ylabel("Percentage of decayed samples")
plt.title("Percentage of decayed samples in each cluster")
plt.show()

for i in range(N_CLUSTERS):
    print(f'# of samples in cluster {i}: {relevant_cluster_counts[i]}')

# %%
j = 6
random.seed(42)
rc = relevant_captions[relevant_cluster_assignments==j]
ru = relevant_urls[relevant_cluster_assignments==j]
rands = random.sample(range(len(rc)), min(50, len(rc)))
for i in rands:
    print(rc[i])
    print(ru[i])
    print()
# %%

# %%

# %%

# %%

# %%























# %%
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
# %%
""" text = ["image may contain : person , playing a sport , on stage , basketball court , shoes and outdoor"] """
""" text = ["This is a gitl with heart on sleeve shirt and tank top"] """
text = ["This is a community event with students and veterans",
        "This is a black and white scarface poster",
        "This is a meme",
        "This is a gun meme",
        "This is a black and white movie poster",
        "This is a black and white poster"]
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
k = 0
chosen = np.where(txt_vs_embeddings[k] > 0.8)[0]
relevant_cc_embeddings = cc_embeddings[chosen]
relevant_captions = captions[chosen]
relevant_urls = url[chosen]
relevant_decay_positions = np.where(decayed_array[chosen]==1)

random.seed(42)
rands = random.sample(range(len(relevant_captions)), min(50, len(relevant_captions)))
for i in rands:
    if decayed_array[chosen][i] == 1:
        print("DECAYED")
    else:
        print("EXISTING")
    print(relevant_captions[i])
    print(relevant_urls[i])
    print()
# %%

# %%

# %%
