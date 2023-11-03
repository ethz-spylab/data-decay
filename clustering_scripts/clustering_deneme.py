#%%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
IMAGENET_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "imagenet_class_embeddings_L14.npy"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
# CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings.npy"
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
CC_VS_IMAGENET = EMBEDDINGS_FOLDER / "CC_vs_imagenet_L14.npy"
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
    plot_imagenet_make_up, find_matching_labels_and_clusters,get_label_cluster_matching_captions_urls)
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

IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"

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

#%%

#dot_products = 1 - distances


# %%
# TODO: check if dot products make sense
get_relevant_captions_and_urls(dot_products, 30, only_argmax=True, sort_best=False)

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
captions = np.array(cc_captions["caption"])
url = np.array(cc_captions["url"])

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
#cc_vs_imagenet = np.load(CC_VS_IMAGENET)
# %%
#cc_vs_imagenet.shape
# %%
#imagenet_assignment = np.argmax(cc_vs_imagenet, axis=1)
#imagenet_assignment.shape

# %%
#np.save(CC_VS_IMAGENET_ASSIGNMENT, imagenet_assignment)
imagenet_assignment = np.load(CC_VS_IMAGENET_ASSIGNMENT)

# %%
imagenet_assignment.shape


# %%
# TODO: find percentage of decayed indices in each imagenet label
decayed_in_imagenet = np.zeros(IMAGENET_LABEL_COUNT, dtype=int)
for i in range(IMAGENET_LABEL_COUNT):
    decayed_in_imagenet[i] = np.count_nonzero(imagenet_assignment[decayed_indices] == i)
percentage_of_decayed_indices_in_imagenet = decayed_in_imagenet / np.sum(decayed_in_imagenet)


# %%
# TODO: find how many elements are there in each imagenet assignment
imagenet_element_counts = np.zeros(IMAGENET_LABEL_COUNT)
for i in range(IMAGENET_LABEL_COUNT):
    imagenet_element_counts[i] = np.count_nonzero(imagenet_assignment == i)


# TODO: find the number of imagenet_element_counts with 0 elements
print("Number of imagenet_element_counts with 0 elements: ", np.count_nonzero(imagenet_element_counts == 0))


# %%

plot_cluster_make_up(68, cluster_assignment, imagenet_assignment, 
                    decayed_indices, imagenet_classes_short,
                    imagenet_element_count_threshold = 300, order="percentage_of_decayed")

# %%

for i in highest_number_clusters:
    plot_cluster_make_up(i, cluster_assignment, imagenet_assignment, 
                        decayed_indices, imagenet_classes_short,
                        imagenet_element_count_threshold = 100, order="percentage_of_decayed")
    

# %%

for i in highest_percentage_clusters:
    plot_cluster_make_up(i, cluster_assignment, imagenet_assignment, 
                        decayed_indices, imagenet_classes_short,
                        imagenet_element_count_threshold = 100, order="percentage_of_decayed")
    

# %%

for i in highest_percentage_clusters:
    plot_cluster_make_up(i, cluster_assignment, imagenet_assignment, 
                        decayed_indices, imagenet_classes_short,
                        imagenet_element_count_threshold = 100, order="number")


    
# %%

relevant_labels, relevant_clusters = find_matching_labels_and_clusters(cluster_assignment,
                                         imagenet_assignment, decayed_indices,
                                         imagenet_classes_short, imagenet_classes_long,
                                         imagenet_element_count_threshold  = 1000,
                                         imagenet_percentage_in_cluster_threshold  = 0.3,
                                         cluster_percentage_in_imagenet_threshold  = 0.3,
                                         decay_percentage_of_label_in_cluster_threshold = 0.15)


# %%

relevant_captions, relevant_urls = \
    get_relevant_captions_and_urls(dot_products, relevant_clusters[5], only_argmax=True, sort_best=False)

# %%
print(*relevant_captions, sep="\n")
print(*relevant_urls, sep="\n")



# %%


label_cluster_cap , label_cluster_url = get_label_cluster_matching_captions_urls(imagenet_assignment, 
                                                                                 cluster_assignment, 
                                                                                 relevant_labels[2], 
                                                                                 relevant_clusters[2])

# %%

print(*label_cluster_cap, sep="\n")
print(*label_cluster_url, sep="\n")

# %%

a,b=get_relevant_captions_and_urls(dot_products, relevant_clusters[3], only_argmax=True, sort_best=False)

print(*a, sep="\n")
print(*b, sep="\n")

# %%

# For cluster 82 / 100
relevant_cluster = 82
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:0"
model.to(device)
text = ["This is a market.","This is a market place.",
        "This is a store.","This is a supermarket.",
        "This is a grocery store.","This is a grocery.",
        "This is a grocery shop.","This is a grocery market.",
        "This is a food market.", "This is a food store.", "This is a food shop."]
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

label_vs_cluster = imagenet_label_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {imagenet_classes_short[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster):.3f}')

# %%

# For cluster 35 / 100
relevant_cluster = 35
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:0"
model.to(device)
text = ["This is a team sport.","This is a team game.",
        "This is a team.","This is a team activity.",
        "This is a team event.","This is a team competition.",
        "This is a celebration.","This is a team celebration.",
        "This is a team party.","This is a team gathering.",
        "This is a team celebrating a victory.","This is a team celebrating a win.",
        "This is a victory celebration.","This is a win celebration."]
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

label_vs_cluster = imagenet_label_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {imagenet_classes_short[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster):.3f}')

# %%

# For cluster 386 / 500
relevant_cluster = relevant_clusters[2]
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
text = ["This is a ice hockey.","This is a hockey.",
        "This is a ice hockey player.","This is a hockey player.",
        "This is a ice hockey game.","This is a hockey game.",
        "This is a ice hockey match.","This is a hockey match.",
        "This is a ice", "This is a athlete", "This is a olympic hockey."]
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

label_vs_cluster = imagenet_label_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {imagenet_classes_short[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster):.3f}')

# %%

# For cluster 405 / 500
relevant_cluster = relevant_clusters[3]
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
text = ["This is a bathroom.","This is a toilet.",
        "This is a restroom.","This is a washroom.",
        "This is a laundry room.","This is a washbasin.",
        "This is a laundry.","This is a bathroom sink.",
        "This is a sink.", "This is a shower.", "This is a bathtub."]
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
            \n similarity: {txt_vs_cluster[i]}')


sargmax = np.argmax(txt_vs_cluster)
print(f'Most similar description to cluster center: {text[sargmax]}, \
      \n similarity: {txt_vs_cluster[sargmax]}')

label_vs_cluster = imagenet_label_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {imagenet_classes_short[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster)}')

# %%

# For cluster 464 / 500
relevant_cluster = relevant_clusters[1]
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
text = ["This is a bathroom.","This is a toilet.",
        "This is a restroom.","This is a washroom.",
        "This is a laundry room.","This is a laundry."]
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
            \n similarity: {txt_vs_cluster[i]}')


sargmax = np.argmax(txt_vs_cluster)
print(f'Most similar description to cluster center: {text[sargmax]}, \
      \n similarity: {txt_vs_cluster[sargmax]}')

label_vs_cluster = imagenet_label_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {imagenet_classes_short[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster)}')


# %%

# For cluster 400 / 500
relevant_cluster = relevant_clusters[0]
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
text = ["This is a graduation.","This is a student.",
        "This is a academic ceremony.",
        "This is a gradution ceremony."]
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
            \n similarity: {txt_vs_cluster[i]}')


sargmax = np.argmax(txt_vs_cluster)
print(f'Most similar description to cluster center: {text[sargmax]}, \
      \n similarity: {txt_vs_cluster[sargmax]}')

label_vs_cluster = imagenet_label_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {imagenet_classes_short[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster)}')


# %%

for i in highest_percentage_clusters:
    plot_cluster_make_up(i, cluster_assignment, imagenet_assignment, 
                        decayed_indices, imagenet_classes_short,
                        imagenet_element_count_threshold = 100, order="number")
    
# %%

a,b=get_relevant_captions_and_urls(dot_products, 
                                   highest_percentage_clusters[6], 
                                   only_argmax=True, 
                                   sort_best=False)

print(*a, sep="\n")
print(*b, sep="\n")

# %%

a , b = get_label_cluster_matching_captions_urls(imagenet_assignment, 
                                                cluster_assignment, 
                                                344, 
                                                highest_percentage_clusters[6])

print(*a, sep="\n")
print(*b, sep="\n")

# %%highest_percentage_clusters


# %%

# For cluster 105 / 500
relevant_cluster = highest_percentage_clusters[6]
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
text = ["This is a horse.","This is a person riding a horse."]
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

label_vs_cluster = imagenet_label_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {imagenet_classes_short[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster):.3f}')

print(f'# of elements in cluster: {int(cluster_element_counts[relevant_cluster])}, \
    \n# of decayed elements in cluster: {int(decayed_in_clusters[relevant_cluster])}, \
    \nDecay rate in cluster: {percentage_of_decayed_indices_in_clusters[relevant_cluster]:.3f}')


# %%

# %%

relevant_labels, relevant_clusters = find_matching_labels_and_clusters(cluster_assignment,
                                         imagenet_assignment, decayed_indices,
                                         imagenet_classes_short, imagenet_classes_long,
                                         imagenet_element_count_threshold  = 1000,
                                         imagenet_percentage_in_cluster_threshold  = 0.3,
                                         cluster_percentage_in_imagenet_threshold  = 0.3,
                                         decay_percentage_of_label_in_cluster_threshold = 0.2)


# %%

a,b=get_relevant_captions_and_urls(dot_products, 
                                   relevant_clusters[6], 
                                   only_argmax=True, 
                                   sort_best=False)

print(*a, sep="\n")
print(*b, sep="\n")

# %%

a , b = get_label_cluster_matching_captions_urls(imagenet_assignment, 
                                                cluster_assignment, 
                                                817, 
                                                120)

print(*a, sep="\n")
print(*b, sep="\n")

# %%

# For cluster 259
relevant_cluster = relevant_clusters[6]
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
text = ["This is a lamp.","This is a light.",
        "This is a lampshade.","This is a lamp shade."]
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

label_vs_cluster = imagenet_label_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {imagenet_classes_short[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster):.3f}')

# %%

# For cluster 162
relevant_cluster = relevant_clusters[5]
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
text = ["This is a car.","This is a vehicle.",
        "This is a automobile."]
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

label_vs_cluster = imagenet_label_embeddings @ cluster_centers[relevant_cluster].T
print(f'Most similar label to cluster center: {imagenet_classes_short[np.argmax(label_vs_cluster)]}, \
      \n similarity: {np.max(label_vs_cluster):.3f}')

# %%
cluster_element_counts[120]











# %%
# TODO: find cluster to caption assignment and print the caption for each cluster

cluster_to_caption_assignment = np.argmax(dot_products, axis=0)
for i in range(number_of_clusters):
    print("Cluster: ", i, " Match: ", 
          dot_products[cluster_to_caption_assignment[i],i],
          " # elements", int(cluster_element_counts[i]),
          " Caption: ", captions[cluster_to_caption_assignment[i]])






#%%

most_decayed = np.argmax(percentage_of_decayed_indices_in_clusters[relevant_clusters])
url_netloc = np.array([urlparse(x).netloc for x in url])
url_netloc_decayed = url_netloc[decayed_indices]
url_netloc_most_decayed_label = url_netloc[imagenet_assignment == relevant_labels[most_decayed]]
url_netloc_most_decayed_label_decayed = url_netloc_decayed[imagenet_assignment[decayed_indices] == relevant_labels[most_decayed]]
url_netloc_most_decayed_cluster = url_netloc[cluster_assignment == relevant_clusters[most_decayed]]
url_netloc_most_decayed_cluster_decayed = url_netloc_decayed[cluster_assignment[decayed_indices] == relevant_clusters[most_decayed]]

#%%

url_netloc_counter = Counter(url_netloc)
url_netloc_decayed_counter = Counter(url_netloc_decayed)
url_netloc_most_decayed_label_counter = Counter(url_netloc_most_decayed_label)
url_netloc_most_decayed_label_decayed_counter = Counter(url_netloc_most_decayed_label_decayed)
url_netloc_most_decayed_cluster_counter = Counter(url_netloc_most_decayed_cluster)
url_netloc_most_decayed_cluster_decayed_counter = Counter(url_netloc_most_decayed_cluster_decayed)

#%%

url_of_interest = url_netloc_most_decayed_label_counter.most_common(1)[0][0]
print("url_of_interest: ", url_of_interest)
print(f'\n# of {url_of_interest} in dataset:  {url_netloc_counter[url_of_interest]}, \
      \n# of those that has decayed:  {url_netloc_decayed_counter[url_of_interest]}, \
      \nwhat % of {url_of_interest} in dataset has decayed:  {url_netloc_decayed_counter[url_of_interest]/url_netloc_counter[url_of_interest]:.3f}')

print(f'\n# of {url_of_interest} in label {relevant_labels[most_decayed]}:  {url_netloc_most_decayed_label_counter[url_of_interest]}, \
      \n# of those that has decayed:  {url_netloc_most_decayed_label_decayed_counter[url_of_interest]}', \
      f'\nwhat % of {url_of_interest} in label {relevant_labels[most_decayed]} has decayed:  {url_netloc_most_decayed_label_decayed_counter[url_of_interest]/url_netloc_most_decayed_label_counter[url_of_interest]:.3f}')

print(f'\n# of {url_of_interest} in cluster {relevant_clusters[most_decayed]}:  {url_netloc_most_decayed_cluster_counter[url_of_interest]}, \
      \n# of those that has decayed:  {url_netloc_most_decayed_cluster_decayed_counter[url_of_interest]}', \
      f'\nwhat % of {url_of_interest} in cluster {relevant_clusters[most_decayed]} has decayed:  {url_netloc_most_decayed_cluster_decayed_counter[url_of_interest]/url_netloc_most_decayed_cluster_counter[url_of_interest]:.3f}')

print(f'\n# of elements in label {relevant_labels[most_decayed]}: {int(imagenet_element_counts[relevant_labels[most_decayed]])}, \
      \n% contribution of {url_of_interest} to label {relevant_labels[most_decayed]}: {url_netloc_most_decayed_label_counter[url_of_interest]/imagenet_element_counts[relevant_labels[most_decayed]]:.3f}, \
      \n# decayed elements in label {relevant_labels[most_decayed]}: {int(decayed_in_imagenet[relevant_labels[most_decayed]])}, \
      \n% of decay in label {relevant_labels[most_decayed]}: {decayed_in_imagenet[relevant_labels[most_decayed]]/imagenet_element_counts[relevant_labels[most_decayed]]:.3f}, \
      \n% decayed contribution of {url_of_interest} to label {relevant_labels[most_decayed]}: {url_netloc_most_decayed_label_decayed_counter[url_of_interest]/decayed_in_imagenet[relevant_labels[most_decayed]]:.3f}')

print(f'\n# of elements in cluster {relevant_clusters[most_decayed]}: {int(cluster_element_counts[relevant_clusters[most_decayed]])}, \
        \n% contribution of {url_of_interest} to cluster {relevant_clusters[most_decayed]}: {url_netloc_most_decayed_cluster_counter[url_of_interest]/cluster_element_counts[relevant_clusters[most_decayed]]:.3f}, \
        \n# decayed elements in cluster {relevant_clusters[most_decayed]}: {int(decayed_in_clusters[relevant_clusters[most_decayed]])}, \
        \n% of decay in cluster {relevant_clusters[most_decayed]}: {decayed_in_clusters[relevant_clusters[most_decayed]]/cluster_element_counts[relevant_clusters[most_decayed]]:.3f}, \
        \n% decayed contribution of {url_of_interest} to cluster {relevant_clusters[most_decayed]}: {url_netloc_most_decayed_cluster_decayed_counter[url_of_interest]/decayed_in_clusters[relevant_clusters[most_decayed]]:.3f}')


#%%

decayed_array = np.zeros(imagenet_assignment.shape[0], dtype=int)
decayed_array[decayed_indices] = 1

for i in range(len(url)):
    if (url_netloc[i] == url_of_interest) & (decayed_array[i] == 0):
        rel_id = i
        print(url[i])

#%%
#Check if those urls are actually decayed
k = url[(url_netloc == url_of_interest) & 
    (imagenet_assignment==relevant_labels[most_decayed]) &
    (cluster_assignment==relevant_clusters[most_decayed]) &
    (decayed_array==1)]

#%%
k[43:47]

#%%
#Randomly sample 10 elements from url[decayed_indices]

np.random.seed(42)
np.random.choice(url[decayed_indices], 10, replace=False)

#%%
decayed_array = np.zeros(imagenet_assignment.shape[0], dtype=int)
decayed_array[decayed_indices] = 1

urls_of_interest_decayed = url[(url_netloc == url_of_interest) &
                               (decayed_array == 1)]
urls_of_interes_nondecayed = url[(url_netloc == url_of_interest) &
                                 (decayed_array == 0)]

#%%
print(np.random.choice(urls_of_interest_decayed, 5, replace=False))

#%%

urls_decayed = url[(decayed_array == 1)]

#%%
print(np.random.choice(urls_decayed, 5, replace=False))


#%%

url_decayed = np.array(url)[decayed_indices]
url_rel = np.array(url)[(imagenet_assignment == relevant_labels[3])
                                & (cluster_assignment == relevant_clusters[3])]

decayed_array = np.ones(imagenet_assignment.shape[0], dtype=int)
decayed_array[decayed_indices] = 0

url_rel_decayed = np.array(url_decayed)[(imagenet_assignment[decayed_indices] == relevant_labels[3])
                                & (cluster_assignment[decayed_indices] == relevant_clusters[3])]



#%%

from urllib.parse import urlparse
from collections import Counter

#%%

url_netloc = [urlparse(x).netloc for x in url]
url_decayed_netloc = [urlparse(x).netloc for x in url_decayed]
url_rel_netloc = [urlparse(x).netloc for x in url_rel]
url_rel_decayed_netloc = [urlparse(x).netloc for x in url_rel_decayed]

#%%

url_netloc_counter = Counter(url_netloc)
url_decayed_netloc_counter = Counter(url_decayed_netloc)
url_rel_netloc_counter = Counter(url_rel_netloc)
url_rel_decayed_netloc_counter = Counter(url_rel_decayed_netloc)
url_rel_name = url_rel_netloc_counter.most_common(1)[0][0]

#%%

print("url_rel_name: ", url_rel_name)
print("url_rel_name in label-cluster count: ", url_rel_netloc_counter[url_rel_name])
print("url_rel_name in label-cluster decayed count: ", url_rel_decayed_netloc_counter[url_rel_name])
print("url_rel_name dataset count: ", url_netloc_counter[url_rel_name])
print("url_rel_name dataset decayed count: ", url_decayed_netloc_counter[url_rel_name])

#%%

for i in range(len(url)):
    if (urlparse(url[i]).netloc == url_rel_name) & (decayed_array[i] == 1):
        print(i)
        print(url[i])

#%%

print(803116 in decayed_indices)
print(803116-2 in decayed_indices)
print(803116+2 in decayed_indices)














#%%

# AFTER THIS POINT IS NOT USED






# %%

[imagenet_classes_short[x]+" ("+str(x)+")" for x in relevant_labels]
print("label: (" + str(relevant_labels[0]) + ") " + imagenet_classes_short[relevant_labels[0]])

#%%

relevant_labels

# %%
#TODO: read IMAGENET_CLASSES_SHORT and save it to a list
imagenet_classes_short = []
with open(IMAGENET_CLASSES_SHORT, "r") as f:
    for line in f:
        imagenet_classes_short.append(line.strip())

imagenet_classes_short = np.array(imagenet_classes_short)

imagenet_classes_long = []
with open(IMAGENET_CLASSES_LONG, "r") as f:
    for line in f:
        imagenet_classes_long.append(line.strip())

imagenet_classes_long = np.array(imagenet_classes_long)

#%%

imagenet_classes_long[relevant_labels][0]






































# %%

# %%
# TODO: find the number of elements in each imagenet label and cluster

imagenet_cluster_counts = np.zeros((IMAGENET_LABEL_COUNT,number_of_clusters), dtype=int)
decayed_imagenet_cluster_counts = np.zeros((IMAGENET_LABEL_COUNT,number_of_clusters), dtype=int)
for i in range(len(cluster_assignment)):
    imagenet_cluster_counts[imagenet_assignment[i], cluster_assignment[i]] += 1

for i in decayed_indices:
    decayed_imagenet_cluster_counts[imagenet_assignment[i], cluster_assignment[i]] += 1

# %%

decayed_in_imagenet = np.sum(decayed_imagenet_cluster_counts, axis=1)
print(decayed_in_imagenet.shape)

# %%
k = np.sum(imagenet_cluster_counts, axis=0)
l = np.sum(imagenet_cluster_counts, axis=1)
print(np.sum(k == cluster_element_counts))
print(np.sum(l == imagenet_element_counts))

# %%
percentages_of_top_labels_in_clusters = get_percentages_of_topk_labels_in_clusters(imagenet_cluster_counts, 1)
percentages_of_top_labels_in_clusters.shape

# %%
percentages_of_top_clusters_for_labels = get_percentages_of_topk_labels_in_clusters(imagenet_cluster_counts.T, 1)
percentages_of_top_clusters_for_labels.shape

# %%
print("max: ", np.nanmax(percentages_of_top_clusters_for_labels), " argmax: " , np.nanargmax(percentages_of_top_clusters_for_labels))
plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[np.nanargmax(percentages_of_top_clusters_for_labels),:])
plt.title('Distribution of imagenet label ' + str(np.nanargmax(percentages_of_top_clusters_for_labels)) + ' in the clusters')
plt.show()

#%%
large_ones = percentages_of_top_clusters_for_labels[imagenet_element_counts > 10000]
large_indices = get_top_n_indices(large_ones, 3)
rel = [np.where(percentages_of_top_clusters_for_labels == large_ones[x])[0][0] for x in large_indices]

for r in rel:
    plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[r,:])
    plt.title('Distribution of imagenet label ' + str(r) + ' in the clusters')
    plt.show()

# %%
#ust limit de konulabilir mesela suan surekli cluster 74 cikio
#decay olanlari da incele
#imagenet label distribution i function hala getirebiliriz

# %%
inds = get_top_n_indices(percentages_of_top_labels_in_clusters, 3)

for ind in inds:
    plot_cluster_make_up(ind, cluster_assignment, imagenet_assignment, decayed_indices, order="number")

# %%

for ind in inds:
    print("ind: ", ind, "cluster element #: ", cluster_element_counts[ind], " top %: ", percentages_of_top_labels_in_clusters[ind])


# %%

for ind in inds:
    print(np.argmax(imagenet_cluster_counts[:,ind]))
    print(imagenet_cluster_counts[np.argmax(imagenet_cluster_counts[:,ind]),ind])

# %%

distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)
#maxa = np.nanmax(a, axis=1)

#%%
print(distribution_of_imagenet_labels_to_clusters.shape)
print(percentages_of_imagenet_labels_in_clusters.shape)
np.sum(percentages_of_imagenet_labels_in_clusters, axis=1)

#%%

poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > 0.5))
rows = poss[:,0]
cols = poss[:,1]

#%%

imagenet_element_count_threshold = 1000
imagenet_percentage_in_cluster_threshold = 0.5
cluster_percentage_in_imagenet_threshold = 0.4

distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)

im_element_count = np.sum(imagenet_cluster_counts, axis=1)
cl_element_count = np.sum(imagenet_cluster_counts, axis=0)

poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > imagenet_percentage_in_cluster_threshold))
rows = poss[:,0]
cols = poss[:,1]

relevant_labels = []
relevant_clusters = []

for row,col in zip(rows, cols):
    if (im_element_count[row] >= imagenet_element_count_threshold) & (percentages_of_imagenet_labels_in_clusters[col, row] >= cluster_percentage_in_imagenet_threshold):
        relevant_labels.append(row)
        relevant_clusters.append(col)

summary = True

for i in range(len(relevant_clusters)):
    plot_cluster_make_up(relevant_clusters[i], cluster_assignment, imagenet_assignment, decayed_indices, order="number")
    plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[relevant_labels[i],:])
    plt.title('Distribution of imagenet label ' + str(relevant_labels[i]) + ' in the clusters')
    plt.show()
    if summary:
        print("label: ", relevant_labels[i], " cluster: ", relevant_clusters[i])
        print("label in cluster: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
              " total label: ", im_element_count[relevant_labels[i]],
              " label percentage in cluster: ", distribution_of_imagenet_labels_to_clusters[relevant_labels[i],relevant_clusters[i]])
        print("cluster in label: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                " total cluster: ", cl_element_count[relevant_clusters[i]],
                " cluster percentage in label: ", percentages_of_imagenet_labels_in_clusters[relevant_clusters[i],relevant_labels[i]])
    



    

# %%

print(percentages_of_imagenet_labels_in_clusters[79,879])
print(distribution_of_imagenet_labels_to_clusters[879,79])

# %%

percentages_of_imagenet_labels_in_clusters[79,879]
distribution_of_imagenet_labels_to_clusters[879,79]


plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[879,:])
plt.title('Distribution of imagenet label ' + str(879) + ' in the clusters')
plt.show()

plot_cluster_make_up(79, cluster_assignment, imagenet_assignment, decayed_indices, order="number")

# %%

np.sum(np.max(distribution_of_imagenet_labels_to_clusters, axis=1) == percentages_of_top_clusters_for_labels)
# nan is not equal to nan

#%%

np.where(imagenet_element_counts == 0)


# %%
#TODO: plot the distribution of imagenet label i in the clusters
for ind in inds:
    plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[np.argmax(imagenet_cluster_counts[:,ind]),:])
    plt.title('Distribution of imagenet label ' + str(ind) + ' in the clusters')
    plt.show()


# %%
rel
# %%
cluster_number = 10
cluster_indices = np.where(cluster_assignment == cluster_number)[0]
# %%
cluster_indices
# %%

imagenet_element_count_threshold = 1000
imagenet_percentage_in_cluster_threshold = 0.5
cluster_percentage_in_imagenet_threshold = 0.4

distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)

im_element_count = np.sum(imagenet_cluster_counts, axis=1)
cl_element_count = np.sum(imagenet_cluster_counts, axis=0)

poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > imagenet_percentage_in_cluster_threshold))
rows = poss[:,0]
cols = poss[:,1]

relevant_labels = []
relevant_clusters = []


row_counter = 0

for row in rows:
    row_counter += 1
    col_counter = 0
    for col in cols:
        col_counter += 1
        if (im_element_count[row] >= imagenet_element_count_threshold) & (percentages_of_imagenet_labels_in_clusters[col, row] >= cluster_percentage_in_imagenet_threshold):
            print("Adding label: ", row, " to cluster: ", col)
            print("Row counter: ", row_counter, " col counter: ", col_counter)
            relevant_labels.append(row)
            relevant_clusters.append(col)
        else:
            continue
# %%
print(relevant_labels)
print(relevant_clusters)

# %%
rows[76]
# %%
a = [1,2,3,4,5]
b = [1,2,3,4,5]

for x,y in zip(a,b):
    print(x,y)
# %%

def get_distribution_of_imagenet_labels_to_clusters2(imagenet_cluster_counts):
    return imagenet_cluster_counts / np.sum(imagenet_cluster_counts, axis=1, keepdims=True)
# %%
distribution_of_imagenet_labels_to_clusters2 = get_distribution_of_imagenet_labels_to_clusters2(imagenet_cluster_counts)
# %%
distribution_of_imagenet_labels_to_clusters2[879,79]
# %%
np.sum(imagenet_cluster_counts, axis=1, keepdims=True).shape
# %%
np.sum(imagenet_cluster_counts, axis=1).shape


# %%

imagenet_element_count_threshold = 1000
imagenet_percentage_in_cluster_threshold = 0.5
cluster_percentage_in_imagenet_threshold = 0.4

distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)

im_element_count = np.sum(imagenet_cluster_counts, axis=1)
cl_element_count = np.sum(imagenet_cluster_counts, axis=0)

poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > imagenet_percentage_in_cluster_threshold))
rows = poss[:,0]
cols = poss[:,1]

relevant_labels = []
relevant_clusters = []

for row,col in zip(rows, cols):
    if (im_element_count[row] >= imagenet_element_count_threshold) & (percentages_of_imagenet_labels_in_clusters[col, row] >= cluster_percentage_in_imagenet_threshold):
        relevant_labels.append(row)
        relevant_clusters.append(col)

summary = True

for i in range(len(relevant_clusters)):
    plot_cluster_make_up(relevant_clusters[i], cluster_assignment, imagenet_assignment, decayed_indices, order="number")
    plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[relevant_labels[i],:])
    plt.title('Distribution of imagenet label ' + str(relevant_labels[i]) + ' to the clusters')
    plt.show()
    if summary:
        print("label: ", relevant_labels[i], " cluster: ", relevant_clusters[i])
        print("label in cluster: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
              " total label: ", im_element_count[relevant_labels[i]],
              " label percentage in cluster: ", distribution_of_imagenet_labels_to_clusters[relevant_labels[i],relevant_clusters[i]])
        print("cluster in label: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                " total cluster: ", cl_element_count[relevant_clusters[i]],
                " cluster percentage in label: ", percentages_of_imagenet_labels_in_clusters[relevant_clusters[i],relevant_labels[i]])
    

# %%

def get_imagenet_cluster_counts_and_decayed(cluster_assignment: np.ndarray,
                                            imagenet_assignment: np.ndarray,
                                            decayed_indices: list = []):
    
    """For each imagenet labels, count how many times it appears in each cluster and how many of them are decayed.
    Args:
        cluster_assignment: cluster assignment for each CC dataset element.
        imagenet_assignment: imagenet label assignment for each CC dataset element.
        decayed_indices: list of indices of decayed CC dataset elements.
    Returns:
        imagenet_cluster_counts: 2D array of shape (# imagenet labels, # clusters) where each row represents an imagenet label and each column represent a cluster.
        decayed_imagenet_cluster_counts: 2D array of shape (# imagenet labels, # clusters) where each row represents an imagenet label and each column represent a cluster.
            Returned only if decayed_indices is not empty."""

    number_of_clusters = np.max(cluster_assignment) + 1
    imagenet_cluster_counts = np.zeros((IMAGENET_LABEL_COUNT,number_of_clusters), dtype=int)
    for i in range(len(cluster_assignment)):
        imagenet_cluster_counts[imagenet_assignment[i], cluster_assignment[i]] += 1

    if len(decayed_indices) > 0:
        decayed_imagenet_cluster_counts = np.zeros((IMAGENET_LABEL_COUNT,number_of_clusters), dtype=int)
        for i in decayed_indices:
            decayed_imagenet_cluster_counts[imagenet_assignment[i], cluster_assignment[i]] += 1

        return imagenet_cluster_counts, decayed_imagenet_cluster_counts
    
    return imagenet_cluster_counts


#%%
def plot_imagenet_make_up(distribution_to_clusters : np.ndarray,
                          decayed_distribution_to_clusters : np.ndarray,
                          title : str = "",
                          n_plot : int = 10,
                          number_of_clusters : int = 100):
    
    top_indices = get_top_n_indices(distribution_to_clusters, n_plot)
    plt.bar([str(x) for x in top_indices], distribution_to_clusters[top_indices], color='g')
    plt.bar([str(x) for x in top_indices], decayed_distribution_to_clusters[top_indices], color='orange')
    plt.title('Distribution of imagenet label ' + title + ' in the clusters')
    plt.legend(['All', 'Decayed'])
    plt.show()    


#%%
def plot_imagenet_make_up(distribution_to_clusters : np.ndarray,
                          decayed_distribution_to_clusters : np.ndarray,
                          imagenet_label_id : str = "",
                          n_plot : int = 10):
    
    """Plot the distribution of imagenet labels in the clusters.
    Args:
        distribution_to_clusters: 1D array of shape (# clusters) where each element represents the number of imagenet labels in the cluster.
        decayed_distribution_to_clusters: 1D array of shape (# clusters) where each element represents the number of decayed imagenet labels in the cluster.
        title: imagenet label id as string.
        n_plot: number of clusters to plot. In addition to that the rest of the clusters are plotted as one bar.
    """
    
    top_indices = get_top_n_indices(distribution_to_clusters, n_plot)
    X_axis = [str(x) for x in top_indices]
    X_axis.append('Rest')

    plt.bar(X_axis, 
            np.append(distribution_to_clusters[top_indices], 
                              np.sum(distribution_to_clusters) - np.sum(distribution_to_clusters[top_indices])), 
            color='g')
    plt.bar(X_axis,
            np.append(decayed_distribution_to_clusters[top_indices], 
                              np.sum(decayed_distribution_to_clusters) - np.sum(decayed_distribution_to_clusters[top_indices])), 
            color='orange')
    plt.title('Distribution of imagenet label ' + imagenet_label_id + ' in the clusters')
    plt.legend(['All', 'Decayed'])
    plt.show()    


# %%

def find_matching_labels_and_clusters(cluster_assignment: np.ndarray, 
                                      imagenet_assignment: np.ndarray,
                                      decayed_indices: list,
                                      imagenet_element_count_threshold : int = 1000,
                                      imagenet_percentage_in_cluster_threshold : float = 0.5,
                                      cluster_percentage_in_imagenet_threshold : float = 0.4,
                                      summary : bool = True):
    
    """Find imagenet labels that are overrepresented in clusters and clusters that are overrepresented in imagenet labels.
    Args:
        cluster_assignment: cluster assignment for each CC dataset element.
        imagenet_assignment: imagenet label assignment for each CC dataset element.
        decayed_indices: list of indices of decayed CC dataset elements.
        imagenet_element_count_threshold: imagenet labels with less than this number of elements are ignored.
        imagenet_percentage_in_cluster_threshold: imagenet labels with less than this percentage in a cluster are ignored.
        cluster_percentage_in_imagenet_threshold: clusters with less than this percentage in an imagenet label are ignored.
        summary: if True, print a summary of the results.
    """
    
    imagenet_cluster_counts, decayed_imagenet_cluster_counts = get_imagenet_cluster_counts_and_decayed(cluster_assignment, imagenet_assignment, decayed_indices)

    distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
    percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)

    im_element_count = np.sum(imagenet_cluster_counts, axis=1)
    cl_element_count = np.sum(imagenet_cluster_counts, axis=0)

    poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > imagenet_percentage_in_cluster_threshold))
    rows = poss[:,0]
    cols = poss[:,1]

    relevant_labels = []
    relevant_clusters = []

    for row,col in zip(rows, cols):
        if (im_element_count[row] >= imagenet_element_count_threshold) & (percentages_of_imagenet_labels_in_clusters[col, row] >= cluster_percentage_in_imagenet_threshold):
            relevant_labels.append(row)
            relevant_clusters.append(col)

    for i in range(len(relevant_clusters)):
        plot_cluster_make_up(relevant_clusters[i], cluster_assignment, imagenet_assignment, decayed_indices, order="number")
        plot_imagenet_make_up(imagenet_cluster_counts[relevant_labels[i],:], decayed_imagenet_cluster_counts[relevant_labels[i],:], title=str(relevant_labels[i]))
        if summary:
            print("label: ", relevant_labels[i], " cluster: ", relevant_clusters[i])
            print("label in cluster: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                  " total label: ", im_element_count[relevant_labels[i]],
                  " label percentage in cluster: ", distribution_of_imagenet_labels_to_clusters[relevant_labels[i],relevant_clusters[i]])
            print("cluster in label: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                  " total cluster: ", cl_element_count[relevant_clusters[i]],
                  " cluster percentage in label: ", percentages_of_imagenet_labels_in_clusters[relevant_clusters[i],relevant_labels[i]])
        


# %%

find_matching_labels_and_clusters(cluster_assignment, imagenet_assignment, decayed_indices)

# %%

relevant_label = 20
plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[relevant_label,:])
plt.title('Distribution of imagenet label ' + str(relevant_label) + ' in the clusters')
plt.show()

#%%
a = []
len(a)
# %%
a = [1,2,3]
a.append(4)
a
# %%
a = np.zeros(5)
np.append(a, 1)

# %%
#cc_vs_imagenet.shape
# %%
#rewards = cc_vs_imagenet
rewards = 0
# %%
def only_max_similarity_count(rewards):
    return np.sum((rewards == rewards.max(axis=0, keepdims=1)).astype(float), axis=0)

# %%
imagenet_rewards =  only_max_similarity_count(rewards)
# %%
imagenet_rewards_missing = only_max_similarity_count(rewards[decayed_indices,:])
# %%
imagenet_rewards_missing
# %%
imagenet_rewards_missing_percentage = imagenet_rewards_missing / imagenet_rewards
# %%
imagenet_rewards_missing_percentage
# %%
np.max(imagenet_rewards_missing_percentage)
# %%

a = np.array([[1,9,3],[7,5,6]])
a = np.array([1,9,3])
b = (a == a.max(axis=0, keepdims=1)).astype(float)
b


# %%
np.sum((rewards == rewards.max(axis=0, keepdims=1)).astype(float), axis=0)

# %%
def reward_function(reward_row):
    return (reward_row == reward_row.max(keepdims=1))
# %%
imagenet_rewards = rewards
for i in tqdm(range(rewards.shape[0])):
    imagenet_rewards[i,:] = reward_function(rewards[i,:])


# %%
rewards.shape[0]
# %%
imagenet_rewards.shape
# %%
np.sum(np.sum(imagenet_rewards, axis=1) == 1)

# %%
k = np.where(np.sum(imagenet_rewards, axis=1) != 1)
# %%
k
# %%
imagenet_rewards[k[0]]
# %%
np.where(imagenet_rewards[k[0]] != 0)
# %%
poss = np.column_stack(np.where(imagenet_rewards[k[0]] != 0))
rows = poss[:,0]
cols = poss[:,1]
# %%
rows
# %%
np.sum((rewards[rows[0]] == rewards[rows[0]].max(keepdims=1)))
# %%
print(rewards[rows[0], cols[0]])
print(rewards[rows[1], cols[1]])
# %%
np.sum(np.isnan(rewards))
# %%
(rewards[rows[0]] == rewards[rows[0]].max(keepdims=1))
# %%
rewards
# %%
#cc_vs_imagenet
# %%
a = np.array([1,2,3])
b = a
b[0] = 5
# %%
a
# %%
a = np.array([1,2,3])
b = np.copy(a)
b[0] = 5
# %%
a
# %%
min(3,5)
# %%
a = np.array([1000,1002,1003])
fig, ax1 = plt.subplots()
ax1.set_yscale('log')
ax1.bar(np.arange(3), a)
ax1.set_ylim(ymin=1)
plt.show()

# %%
a = np.array([1,2,3])
b = np.array([4,5,6])*10
# %%
str("%.2f" % round(a/b, 2))
# %%
a/b
# %%
str("%.2f" % round(a[0]/b[0]*100, 2))+" %"
# %%
