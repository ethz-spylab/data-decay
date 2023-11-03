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

from tqdm import tqdm
from urllib.parse import urlparse
from collections import Counter
import random

from sklearn.cluster import KMeans, MiniBatchKMeans
import os

IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"

CC_VS_CSAIL_PLACES = EMBEDDINGS_FOLDER / "CC_vs_csail_places.npy"
CC_VS_CSAIL_PLACES_FAST = EMBEDDINGS_FOLDER / "CC_vs_csail_places_fast.npy"
CC_VS_CSAIL_PLACES_ASSIGNMENT = EMBEDDINGS_FOLDER / "CC_vs_csail_places_assignment.npy"
CSAIL_PLACES = DATA_FOLDER / "csail_places.txt"

""" IMAGENET_EMBEDDINGS_FOLDER = CC_VS_CSAIL_PLACES
IMAGENET_CLASSES_SHORT = CSAIL_PLACES
IMAGENET_CLASSES_LONG = CSAIL_PLACES
IMAGENET_LABEL_COUNT = CSAIL_PLACES_COUNT
CC_VS_IMAGENET = CC_VS_CSAIL_PLACES
CC_VS_IMAGENET_FAST = CC_VS_CSAIL_PLACES_FAST """

from transformers import CLIPProcessor, CLIPModel
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
# TODO: check if dot products make sense
# get_relevant_captions_and_urls(dot_products, 30, only_argmax=True, sort_best=False)

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
""" flattened = np.reshape(cc_vs_imagenet,(-1))
plt.hist(flattened, bins=100)
plt.xlabel("Similarity score")
plt.ylabel("Count")
plt.title("CC vs imagenet similarity scores")
plt.show()

print(f'mean: {np.mean(flattened):.2f}, \
      \nstdev: {np.std(flattened):.2f}') """
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
k = 10
highest_percentage = get_top_n_indices(percentage_decayed_in_imagenet, k)
fig = plot_missing_num_perc(highest_percentage, 
                      cc_distribution_to_imagenet,
                      percentage_decayed_in_imagenet,
                      labels = [imagenet_classes_short[x] for x in highest_percentage])

plt.show()
# %%
k = 10
num_threshold = 50
percentage_decayed_in_imagenet_cp = percentage_decayed_in_imagenet.copy()
percentage_decayed_in_imagenet_cp[cc_distribution_to_imagenet < num_threshold] = 0
highest_percentage_num_threshold = get_top_n_indices(percentage_decayed_in_imagenet_cp, k)
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
highest_decayed = get_top_n_indices(decayed_cc_distribution_to_imagenet, k)
fig = plot_missing_num_perc(highest_decayed, 
                      cc_distribution_to_imagenet,
                      percentage_decayed_in_imagenet,
                      labels = [imagenet_classes_short[x] for x in highest_decayed])

plt.show()
# %%
imagenet_classes_long[get_top_n_indices(decayed_cc_distribution_to_imagenet, k)[0]]
# %%
# TODO: test a text match
highest_to_plot = get_top_n_indices(percentage_decayed_in_imagenet_cp, 200)
text = [imagenet_classes_long[x] for x in highest_to_plot]
txt_embeds = imagenet_label_embeddings[highest_to_plot]

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

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
""" text = ["This is a sports car", "This is a vintage car",
        "This is a classic car","This is a automobile"] """
""" text = ["This is a fox","This is a gray fox"] """
""" text = ["This is a bride", "This is a groom",
        "This is a wedding", "This is a marriage",
        "This is a wedding dress", "This is a wedding cake",
        "This is a bride and groom", "This is a wedding ceremony",
        "This is a wedding ring", "This is a wedding reception"] """
""" text = ["This is a bed", "This is a four-poster bed","This is a four-poster",
        "This is a bedroom", "This is a bedroom with a bed",
        "This is a bedroom with a four-poster bed"] """
""" text = ["This is a front grille of a car","This is a radiator grille of a car",
        "This is a car radiator grille"] """
""" text = ["This is a bride walking down the aisle","This is a father walking his daughter down the aisle",
        "This is a father walking bride down the aisle",
        "This is a bride and groom",
        "This is a bride and groom during wedding ceremony",
        "This is a groom and bride with their family ",
        "This is a groom and bride with their family taking a picture",
        "This is a bride and groom during wedding reception"] """
""" text = ["This is a lobby", "This is a hotel lobby","This is a hotel reception",
        "This is a hotel", "This is a artistical rendering of a lobby",
        "This is a rendering of a lobby", "This is a rendering of a hotel lobby",
        "This is a rendering of a hotel reception",
        "This is a artistical rendering of a hotel"] """
""" text = ["This is a people in a lobby","This is a lobby with people",
        "This is a hotel lobby with people", "This is a hotel reception with people",
        "This is a people entering a hotel lobby",
        "This is a people entering lobby"] """
""" text = ["This is a loft","This is a attic",
        "This is a apartment"] """
""" text = ["This is a showroom","This is a car showroom",
        "This is a car dealership","This is a car dealer",
        "This is a car exhibition","This is a auto showroom",
        "This is a auto exhibition"] """
text = ["This is a sports car", "This is a vintage car",
        "This is a classic car","This is a automobile",
        "This is a fox","This is a gray fox",
        "This is a bride", "This is a groom",
        "This is a wedding", "This is a marriage",
        "This is a wedding dress", "This is a wedding cake",
        "This is a bride and groom", "This is a wedding ceremony",
        "This is a wedding ring", "This is a wedding reception",
        "This is a bed", "This is a four-poster bed","This is a four-poster",
        "This is a bedroom", "This is a bedroom with a bed",
        "This is a bedroom with a four-poster bed",
        "This is a front grille of a car","This is a radiator grille of a car",
        "This is a car radiator grille",
        "This is a bride walking down the aisle","This is a father walking his daughter down the aisle",
        "This is a father walking bride down the aisle",
        "This is a bride and groom",
        "This is a bride and groom during wedding ceremony",
        "This is a groom and bride with their family ",
        "This is a groom and bride with their family taking a picture",
        "This is a bride and groom during wedding reception",
        "This is a lobby", "This is a hotel lobby","This is a hotel reception",
        "This is a hotel", "This is a artistical rendering of a lobby",
        "This is a rendering of a lobby", "This is a rendering of a hotel lobby",
        "This is a rendering of a hotel reception",
        "This is a artistical rendering of a hotel",
        "This is a people in a lobby","This is a lobby with people",
        "This is a hotel lobby with people", "This is a hotel reception with people",
        "This is a people entering a hotel lobby",
        "This is a people entering lobby",
        "This is a loft","This is a attic",
        "This is a apartment",
        "This is a showroom","This is a car showroom",
        "This is a car dealership","This is a car dealer",
        "This is a car exhibition","This is a auto showroom",
        "This is a auto exhibition",
        "This is a market.","This is a market place.",
        "This is a store.","This is a supermarket.",
        "This is a grocery store.",
        "This is a team celebration.",
        "This is a team party.",
        "This is a team celebrating a victory.",
        "This is a electric guitar","This is a guitar",
        "This is a musical instrument","This is a person playing a guitar",
        "This is a person playing a guitar on stage",
        "This is a person playing a musical instrument",
        "This is a person playing a musical instrument on stage",
        "This is a person playing a electric guitar on a stage",
        "This is a artist playing a electric guitar on a stage",
        "This is a person performing on a stage with a guitar",
        "This is a person performing on a stage with a electric guitar",
        "This is a artist performing on a stage with a electric guitar"]
""" text = ["image may include, person, guitar, stage"] """
""" text = ["This is a disc brake","This is a disk brake",
        "This is a rear wheel","This is a wheel",
        "This is a tire","This is a bike wheel","This is a bike tire",
        "This is a suspension","This is a bike suspension"] """
""" text = ["This is a wolf", "This is a grey wolf",
        "This is a fox", "This is a red fox"] """
""" text = ["This is a white dog"] """
""" text = ["This is a obelisk","This is a monument",
        "This is a tower","This is a statue",
        "This is a sculpture","This is a statue of a person",
        "This is a important landmark"] """
""" text = ["This is a automobile advertisement","This is a car engine",
        "This is a inside of a car","This is a car interior",
        "This is a luxury car"] """
""" text = ["This is a meatloaf","This is a meatballs"] """
""" text = ["This is a pizza","This is a dough",
        "This is a pizza in the oven"] """
""" text = ["This is a action car","This is a supercar"] """
""" text = ["This is a pot","This is a pan","This is a cooking pot",
        "This is a teapot","This is a bowl","This is a hotpot",
        "This is food in bowl"] """
""" text = ["This is a tiger shark","This is a shark",
        "This is a tiger"] """
""" text = ["This is a sunglasses","This is a sunscreen",
        "This is a skin lotion"] """
""" text = ["This is a laptop","This is a computer",
        "This is a notebook computer"] """
""" text = ["This is a penguin"] """
""" text = ["This is a bathtub"] """
""" text = ["This is a whale"] """
""" text = ["This is a belt","This is a seat belt"] """
""" text = ["This is a toilet"] """
""" text = ["This is a koala", "This is a bear",
        "This is a kangaroo", "This is a panda"] """
""" text = ["This is a ballpoint pen"] """
""" text = ["This is a lipstick","This is a lip gloss",
        "This is a woman wearing lipstick","This is a red lipstick",
        "This is a woman wearing makeup"] """
""" text = ["This is a lamp","This is a floor lamp",
        "This is a chandelier"] """
text = ["This is a buddhist temple", "This is a stupa"]
inputs = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
with torch.no_grad():
    outputs = model.text_model(**inputs)
    txt_embeds = outputs[1]
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
txt_embeds = txt_embeds.cpu().numpy().astype('float32')

""" highest_to_plot = get_top_n_indices(percentage_decayed_in_imagenet_cp, 200)
text = [imagenet_classes_long[x] for x in highest_to_plot]
txt_embeds = imagenet_label_embeddings[highest_to_plot] """

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
random.seed(42)
k = 180
selection = highest_to_plot.copy()
relevant_captions = captions[imagenet_to_cc[selection[k]]]
rands = random.sample(range(len(relevant_captions)), 10)
print(*[captions[imagenet_to_cc[selection[k]][x]] for x in rands], sep="\n")
url[imagenet_to_cc[selection[k]][rands]]
# %%
# mesela sadece bu cars icinde clustering yapip yuksek degerli olanlara bak

# %%
k = 163
#selection = highest_percentage_num_threshold.copy()
selection = highest_to_plot.copy()
relevant_cc_embeddings = cc_embeddings[imagenet_to_cc[selection[k]]]
relevant_captions = captions[imagenet_to_cc[selection[k]]]
relevant_urls = url[imagenet_to_cc[selection[k]]]
relevant_decay_positions = [np.where(imagenet_to_cc[selection[k]] == x)[0][0] for x in decayed_imagenet_to_cc[selection[k]]]
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
relevant_captions[relevant_cluster_assignments==4]
# %%
relevant_urls[relevant_cluster_assignments==1]
# %%
relevant_captions
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
