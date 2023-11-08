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


IMAGENET21K_LONG = DATA_FOLDER / "imagenet21k.txt"
IMAGENET21K_SHORT = DATA_FOLDER / "imagenet21k_short.txt"
IMAGENET21K_EMBEDDINGS = EMBEDDINGS_FOLDER / "imagenet21k_embeddings.npy"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
CC_VS_IMAGENET21K_FOLDER = EMBEDDINGS_FOLDER / "CC_vs_imagenet21k"
CC_VS_IMAGENET21K_COUNT = 22

IMAGENET21K_TO_CC = CC_VS_IMAGENET21K_FOLDER / "imagenet21k_to_cc.txt"
CC_TO_IMAGENET21K = CC_VS_IMAGENET21K_FOLDER / "cc_to_imagenet21k.txt"
DECAYED_IMAGENET21K_TO_CC = CC_VS_IMAGENET21K_FOLDER / "decayed_imagenet21k_to_cc.txt"
DECAYED_CC_TO_IMAGENET21K = CC_VS_IMAGENET21K_FOLDER / "decayed_cc_to_imagenet21k.txt"


IMAGENET_CLASSES_SHORT = IMAGENET21K_SHORT
IMAGENET_CLASSES_LONG = IMAGENET21K_LONG
IMAGENET_LABEL_COUNT = IMAGENET21K_LABEL_COUNT
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
cc_embeddings = fast_load(str(CC_EMBEDDINGS_FOLDER))
# %%
cc_vs_imagenet = np.zeros((len(cc_embeddings),len(imagenet_classes_short)), dtype=np.float16)
for i in tqdm(range(CC_VS_IMAGENET21K_COUNT)):
    cc_vs_imagenet[:,i*STEP_SIZE:(i+1)*STEP_SIZE] = fast_load(str(CC_VS_IMAGENET21K_FOLDER / f"CC_vs_imagenet21k_{i}.npy"))
# %%
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
# %%
""" imagenet_to_cc = []
for i in tqdm(range(IMAGENET_LABEL_COUNT)):
    imagenet_to_cc.append(np.where(cc_vs_imagenet[:, i] > threshold)[0])
imagenet_to_cc2 = [imagenet_to_cc[i].tolist() for i in range(len(imagenet_to_cc))]
with open(IMAGENET21K_TO_CC, 'w') as fp:
    json.dump(imagenet_to_cc2, fp) """

with open(IMAGENET21K_TO_CC) as fp:
    imagenet_to_cc = json.load(fp)
imagenet_to_cc = [np.array(imagenet_to_cc[i], dtype=int) for i in range(len(imagenet_to_cc))]
# %%
""" cc_to_imagenet = []
for i in tqdm(range(cc_vs_imagenet.shape[0])):
    cc_to_imagenet.append(np.where(cc_vs_imagenet[i, :] > threshold)[0])
cc_to_imagenet2 = [cc_to_imagenet[i].tolist() for i in range(len(cc_to_imagenet))]
with open(CC_TO_IMAGENET21K, 'w') as fp:
    json.dump(cc_to_imagenet2, fp) """

with open(CC_TO_IMAGENET21K) as fp:
    cc_to_imagenet = json.load(fp)
cc_to_imagenet = [np.array(cc_to_imagenet[i], dtype=int) for i in range(len(cc_to_imagenet))]
# %%
decayed_cc_to_imagenet = [cc_to_imagenet[i] for i in decayed_indices]
# %%
""" decayed_imagenet_to_cc = []
decayed_cc_vs_imagenet = cc_vs_imagenet[decayed_indices]
for i in tqdm(range(IMAGENET_LABEL_COUNT)):
    decayed_imagenet_to_cc.append([decayed_indices[x] for x in np.where(decayed_cc_vs_imagenet[:, i] > threshold)[0]])
with open(DECAYED_IMAGENET21K_TO_CC, 'w') as fp:
    json.dump(decayed_imagenet_to_cc, fp) """

with open(DECAYED_IMAGENET21K_TO_CC) as fp:
    decayed_imagenet_to_cc = json.load(fp)
decayed_imagenet_to_cc = [np.array(decayed_imagenet_to_cc[i], dtype=int) for i in range(len(decayed_imagenet_to_cc))]
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
k = 200
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

""" text = ["This is a dog sitter", "This is a pet sitter",
        "This is a person sitting next to a dog", "This is a person sitting next to a pet",
        "This is a person feeding a dog", "This is a person feeding a pet",
        "This is a person with a dog", "This is a person with a pet",
        "This is a person walking a dog","This is a person walking a pet"] """
""" text = ["This is a frog", "This is a starfish"] """
""" text = ["This is a capacidance", "This is a resistor",
        "This is a set of electronic components"] """
""" text = ["This is a buffalo", "This is a sculpture of a buffalo",
        "This is a buffalo sculpture"] """
""" text = ["This is a brownie", "This is cocoa powder",
        "This is a cup of hot cocoa","This is a cup of hot chocolate",
        "This is a cup of hot coffee","This is a cup of hot tea"] """
""" text = ["This is a shrine", "This is a funeral",
        "This is a cemetery","This is a graveyard",
        "This is a tomb","This is a tombstone",
        "This is a funeral home","This is a funeral parlor",
        "This is a mortuary","This is a mausoleum"] """
""" text = ["This is a real estate agent"] """
""" text = ["This is a samoyed puppy"] """
""" text = ["This is flour", "This is a bag of flour",
        "This is egg spilled on the flour", "This is a broken egg",
        "This is a omellete", "This is a fried egg"] """
""" text = ["This is a meringue"] """
""" text = ["This is a microchip", "This is a computer chip",
        "This is a chip", "This is a microprocessor",
        "This is a graphics card"] """
""" text = ["This is a house plan", "This is a building plan",
        "This is a home decoration","This is a room decoration",
        "This is a living room decoration","This is a bedroom decoration",
        "This is a office furniture", "This is a light bulb",
        "This is a large house", "This is a mansion"] """
""" text = ["This is a gun", "This is a rifle",
        "This is a pistol", "This is a shotgun",
        "This is a thief", "This is a robber",
        "This is a burglar", "This is a criminal",
        "This is a police officer", "This is a cop",
        "This is a police car", "This is a police vehicle"] """
""" text = ["This is a bass fish", "This is a yellow bass fish",
        "This is a red bass"] """
""" text = ["This is a rhododendron", "This is rhododendrons"] """
""" text = ["This is a soup", "This is a curry",
        "This is a chicken curry", "This is a chicken dish"] """
""" text = ["This is a drum", "This is a drum set",
        "This is a drum kit", "This is a marching band",
        "This is a kettledrum"] """
""" text = ["This is powder", "This is white powder",
        "This is gun powder", "This is cocaine"] """
""" text = ["This is a nut", "This is chestnut",
        "This is a walnut", "This is a peanut"] """
""" text = ["This is a loft"] """
""" text = ["This is a bunk bed", "This is a bed in a room"] """
""" text = ["This is batter"] """
""" text = ["This is a potage", "This is potage","This is a pottage"] """
""" text = ["This is a home for sale", "This is a house for sale"] """
""" text = ["This is a monkey"] """
""" text = ["This is a suv", "This is a sport utility vehicle",
        "This is a jeep", "This is a land rover",
        "This is a pickup truck", "This is a minivan"] """
""" text = ["This is a writing desk"] """
""" text = ["This is laundry", "This is a laundry basket",
        "This is laundry detergent", "This is a washing machine",
        "This is laundry powder", "This is a laundry soap",
        "This is toilet powder", "This is bath powder"] """
""" text = ["This is a bottle of wine", "This is a bottle of beer",
        "This is a bootle of gin", "This is a bottle of vodka"] """
""" text = ["This is a pine"]
text = ["This is a christmas tree", "This is a christmas decoration",
        "This is a person planting a tree"] """
""" text = ["This is a man shaving his beard", "This is a barber"] """
""" text = ["This is a missile", "This is a rocket"] """
""" text = ["This is a oyster", "This is a clam"] """
""" text = ["This is a calculator", "This is a abacus",
        "This is a clerk", "This is a cashier",
        "This is a office worker", "This is a accountant"] """
""" text = ["This is a bunk bed", "This is a feed bunk",
        "This is a bunk"] """
""" text = ["This is a trowel", "This is a shovel",
        "This is a spade", "This is a garden tool",
        "This is a man holding a trowel"] """
""" text = ["This is a shirt tied in the back"] """
""" text = ["This is a priest", "This is a pastor",
        "This is a pope"] """
""" text = ["This is a wizard", "This is a magician"] """
""" text = ["This is a cereal", "This is a cereal oat"] """
""" text = ["This is a hatchback", "This is a hatchback car",
        "This is a hatchback door"] """
""" text = ["This is a blue bottle", "This is a kitten licking a golden retriever"] """
""" text = ["This is a electric guitar","This is a guitar",
        "This is a musical instrument","This is a person playing a guitar",
        "This is a person playing a guitar on stage",
        "This is a person playing a musical instrument",
        "This is a person playing a musical instrument on stage",
        "This is a person playing a electric guitar on a stage",
        "This is a artist playing a electric guitar on a stage",
        "This is a person performing on a stage with a guitar",
        "This is a person performing on a stage with a electric guitar",
        "This is a artist performing on a stage with a electric guitar",
        "image may contain: artist, stage, guitar",
        "image may contain: person, stage, guitar",
        "image may contain : person , on stage and playing a musical instrument",
        "person , on stage and playing a musical instrument",
        "This is a person , on stage and playing a musical instrument"] """
""" text = ["This is a four-poster",
        "This is a four-poster bed", "This is a bed with a canopy",
        "This is a bunk", "This is a bunk bed"] """
""" text = ["This is a radiator","This is a radiator grille",
        "This is a car radiator","This is a car radiator grille",
        "This is grille"] """
""" text = ["This is a microwave oven","This is a microwave"] """
""" text = ["This is a limousine","This is a limo"] """
""" text = ["This is a rendering of a lobby",
        "This is a artistic rendering of a lobby"] """
""" text = ["This is a hotpot","This is a hot pot"] """
""" text = ["This is a laptop, laptop computer"] """
""" text = ["This is a laptop with logo on screen", "This is a laptop"] """
""" text = ["This is a book cover"] """
""" text = ["This is a lipstick", "This is a lip gloss"] """
""" text = ["This is a microprocessor"] """
""" text = ["This is a bath powder", "This is a dusting powder"] """
text = ["This is a wedding"]
text = ["This is a winter wedding", "This is a bride giving a speech",
        "This is a wedding couple walking down the aisle",
        "This is a bride and groom walking down the aisle",
        "This is a father walking bride down the aisle"]
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
k = 0
chosen = np.where(txt_vs_embeddings[k] > 0.5)[0]
relevant_cc_embeddings = cc_embeddings[chosen]
relevant_captions = captions[chosen]
relevant_urls = url[chosen]
relevant_decay_positions = np.where(decayed_array[chosen]==1)
# %%
N_CLUSTERS = 50
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
j = 4
random.seed(42)
rc = relevant_captions[relevant_cluster_assignments==j]
ru = relevant_urls[relevant_cluster_assignments==j]
rands = random.sample(range(len(rc)), min(50, len(rc)))
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
k = 195
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
k = 198
#selection = highest_percentage_num_threshold.copy()
selection = highest_percentage_num_threshold.copy()
relevant_cc_embeddings = cc_embeddings[imagenet_to_cc[selection[k]]]
relevant_captions = captions[imagenet_to_cc[selection[k]]]
relevant_urls = url[imagenet_to_cc[selection[k]]]
relevant_decay_positions = [np.where(imagenet_to_cc[selection[k]] == x)[0][0] for x in decayed_imagenet_to_cc[selection[k]]]
# %%
N_CLUSTERS = 3
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
j = 1
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
rel_dec = np.ones(len(relevant_cluster_assignments))
rel_dec[relevant_decay_positions] = 0

l = 1
relevant_urls[relevant_cluster_assignments==l][rel_dec[relevant_cluster_assignments==l]==1]
# %%
# %%
relevant_captions[relevant_cluster_assignments==2]
# %%
relevant_urls[relevant_cluster_assignments==2]




# %%
# Check the imagenet_label distribution of guitar related captions
cc_vs_imagenet_assignment = np.load(CC_VS_IMAGENET_ASSIGNMENT)
IMAGENET_CLASSES_SHORT_1K = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG_1K = DATA_FOLDER / "imagenet_classes_long.txt"
imagenet_classes_short_1k = []
imagenet_classes_long_1k = []
with open(IMAGENET_CLASSES_SHORT_1K, "r") as f:
    for line in f:
        imagenet_classes_short_1k.append(line.strip())
with open(IMAGENET_CLASSES_LONG_1K, "r") as f:
    for line in f:
        imagenet_classes_long_1k.append(line.strip())
# %%
j = 0
most_common = 3
imagenet_assignment_of_guitar = cc_vs_imagenet_assignment[txt_vs_embeddings[j] > 0.5]
guitar_counter = Counter(imagenet_assignment_of_guitar)
print([imagenet_classes_short_1k[guitar_counter.most_common(most_common)[x][0]] 
 for x in range(most_common)])

# %%
# %%


# %%
c1 = "image may contain : person , on stage and playing a musical instrument"
t1 = "This is a electric guitar"
t2 = "This is a stage"
# %%
inputs = processor(text=[c1,t1,t2], return_tensors="pt", padding=True,truncation=True).to(device)
with torch.no_grad():
    outputs = model.text_model(**inputs)
    txt_embeds = outputs[1]
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
txt_embeds = txt_embeds.cpu().numpy().astype('float32')
# %%
txt_embeds[0] @ txt_embeds[1:].T











# %%
cc_vs_imagenet = np.load(EMBEDDINGS_FOLDER / "CC_vs_imagenet_L14.npy")
# %%
decayed_indices = []
with open(DECAYED_INDICES, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))
print("Number of decayed indices: ", len(decayed_indices))
decayed_array = np.zeros(cc_vs_imagenet.shape[0])
decayed_array[decayed_indices] = 1
# %%
cc_decayed = cc_vs_imagenet[decayed_array==1]
cc_existing = cc_vs_imagenet[decayed_array==0]
# %%
flattened_decayed = np.reshape(cc_decayed,(-1))
flattened_existing = np.reshape(cc_existing,(-1))
# %%
plt.hist(flattened_decayed, bins=100,alpha=0.6, range=(-0.2,1))
plt.hist(flattened_existing, bins=100,alpha=0.6, range=(-0.2,1))
plt.legend(["Decayed", "Existing"])
plt.xlabel("Similarity score")
plt.ylabel("Count")
plt.title("CC vs imagenet similarity scores")
plt.show()