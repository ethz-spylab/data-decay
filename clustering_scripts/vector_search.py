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
    fast_load, fast_save)

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
""" 
clusters_decayed = []
clusters_exist = []
exists_decayed = np.ones(len(cluster_assignment))
exists_decayed[decayed_indices] = 0
for i in range(cluster_counts):
    clusters_decayed.append(np.where((cluster_assignment == i) & (exists_decayed == 0))[0])
    clusters_exist.append(np.where((cluster_assignment == i) & (exists_decayed == 1))[0])



diclist = [{'decayed':x,
            'similar_exist':None,
            'similar_scores':None} for x in decayed_indices]

decayed_dict = {}
for i in range(len(decayed_indices)):
    decayed_dict[decayed_indices[i]] = i

for i in tqdm(range(cluster_counts)):
    temp = cc_embeddings[clusters_decayed[i]] @ cc_embeddings[clusters_exist[i]].T
    for j in range(len(clusters_decayed[i])):
        decayed_ind = clusters_decayed[i][j]
        similar_inds = get_top_n_indices(temp[j], 10)
        diclist[decayed_dict[decayed_ind]]['similar_exist'] = clusters_exist[i][similar_inds].tolist()
        diclist[decayed_dict[decayed_ind]]['similar_scores'] = temp[j][similar_inds].tolist()  """
        

# %%

#with open(CC_DECAYED_SIMILARITY_DICT_100, 'w') as fout:
    #json.dump(diclist, fout)

# %%

with open(CC_DECAYED_SIMILARITY_DICT_100, 'r') as fin:
    diclist = json.load(fin)

# %%

similarities = [diclist[i]['similar_scores'][-1] for i in range(len(diclist))]
plt.hist(similarities, bins=100)
plt.xlabel("Similarity score of 10th most similar in cluster")
plt.ylabel("Count")
plt.show()
# %%
k = np.linspace(0, 1, 100)
dist = [np.sum(np.array(similarities) < i)/len(similarities) for i in k]
plt.plot(k, dist)
plt.xlabel("Similarity score of 10th most similar in cluster")
plt.ylabel("%")
plt.title("Distribution of similarity scores")
plt.show()

# %%

decayed_dict = {}
for i in range(len(decayed_indices)):
    decayed_dict[decayed_indices[i]] = i

# %%
#TODO: randomly pick 1000 decayed indices and find their top 10 similar scores
# then check if it matches with the one we got from clustering
""" # %%
sample_count = 1000
np.random.seed(42)
sampled_indices = np.random.choice(decayed_indices, sample_count, replace=False)
diff = []

for i in tqdm(range(sample_count)):
    decayed_ind = sampled_indices[i]
    real_similarities = cc_embeddings[decayed_ind] @ cc_embeddings.T
    real_similarities[decayed_indices] = 0
    real_farthest_ind = get_top_n_indices(real_similarities, 10)[-1]
    real_farthest_score = real_similarities[real_farthest_ind]
    cluster_farthest_score = diclist[decayed_dict[decayed_ind]]['similar_scores'][-1]
    diff.append(real_farthest_score - cluster_farthest_score)
 """
# %%
#diff_np = np.array(diff)
#np.save("diff2.npy", diff_np)
diff = np.load("diff2.npy")
# %%
plt.hist(diff, bins=100)
plt.xlabel("Difference between real and cluster 10th closest for 1000 random samples")
plt.ylabel("Count")
plt.show()
# %%
# %%
k = np.linspace(0, 1, 100)
dist = [(np.sum(diff < i))/len(diff) for i in k]
plt.plot(k, dist)
plt.xlabel("Difference between real and cluster 10th closest")
plt.ylabel("%")
plt.show()
# %%
print(f'Average difference: {np.mean(diff):.3f}, std: {np.std(diff):.3f}')
# %%
threshold = 0.5
print(f'% decayed indices with similarity less than {threshold}: {np.sum(np.array(similarities) < threshold)/len(similarities)*100:.3f}')
print(f'# decayed indices with similarity less than {threshold}: {np.sum(np.array(similarities) < threshold)}')
# %%
decayed_of_interest = np.where(np.array(similarities) < threshold)[0]
decayed_of_interest = [diclist[x]['decayed'] for x in decayed_of_interest]
decayed_of_interest = np.array(decayed_of_interest)

# %%
""" # TODO: look at those 1000 decayed

diclist[decayed_dict[decayed_of_interest[672]]]
decayed_of_interest_real_scores = []
for i in tqdm(range(len(decayed_of_interest))):
    decayed_ind = decayed_of_interest[i]
    real_similarities = cc_embeddings[decayed_ind] @ cc_embeddings.T
    real_similarities[decayed_indices] = 0
    real_farthest_ind = get_top_n_indices(real_similarities, 10)[-1]
    real_farthest_score = real_similarities[real_farthest_ind]
    cluster_farthest_score = diclist[decayed_dict[decayed_ind]]['similar_scores'][-1]
    decayed_of_interest_real_scores.append(real_farthest_score)

plt.hist(decayed_of_interest_real_scores, bins=100)
plt.xlabel("Real similarity score of 1000 decayed of interest")
plt.ylabel("Count")
plt.show() """
# %%
# TODO: cc_embeddings[decayed_ind] @ cc_embeddings.T yaparken 2.si nondecayed ind icin olmali

# %%
#print(*captions[decayed_of_interest], sep="\n")

# %%
# TODO: test a text match

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:7"
model.to(device)
""" text = ["This is a person on a horse","This is a horse",
        "This is a person smiling","This is a person on a horse smiling",
        "This is a person riding a horse"] """
text = ["This is a electric guitar","This is a guitar",
        "This is a musical instrument","This is a person playing a guitar",
        "This is a person playing a guitar on stage",
        "This is a person playing a musical instrument",
        "This is a person playing a musical instrument on stage",
        "This is a person playing a electric guitar on a stage",
        "This is a artist playing a electric guitar on a stage",
        "This is a person performing on a stage with a guitar",
        "This is a person performing on a stage with a electric guitar",
        "This is a artist performing on a stage with a electric guitar"]
inputs = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
with torch.no_grad():
    outputs = model.text_model(**inputs)
    txt_embeds = outputs[1]
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
txt_embeds = txt_embeds.cpu().numpy().astype('float32')

txt_vs_embeddings = txt_embeds @ cc_embeddings.T

# %%
decayed_txt_vs_embeddings = txt_vs_embeddings[:,decayed_array==1]
decayed_of_interest_txt_vs_embeddings = txt_vs_embeddings[:,decayed_of_interest]
existing_txt_vs_embeddings = txt_vs_embeddings[:,decayed_array==0]
# %%
txt = 7
th = 0.5
plt.hist(decayed_txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
plt.hist(existing_txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
plt.legend(["Decayed", "Existing"])
plt.xlabel("Similarity score")
plt.ylabel("Count")
plt.title("\""+text[txt]+"\""+" vs captions")
plt.show()
# %%
np.random.seed(42)
x = np.where((txt_vs_embeddings[4] > 0.5) &  (decayed_array==0))[0]
x = np.random.choice(x, 10, replace=False)
[imagenet_classes_short[imagenet_assignment[i]] for i in x]
# %%
captions[x]
# %%
cluster_assignment[x]
# %%
np.random.seed(42)
x = np.where((txt_vs_embeddings[4] > 0.4) &  (decayed_array==1))[0]
[imagenet_classes_short[imagenet_assignment[i]] for i in x]
# %%
captions[x]
# %%
cluster_assignment[x]
# %%
threshold = 0.50
print(f'% decayed indices with similarity less than {threshold}: {np.sum(np.array(similarities) < threshold)/len(similarities)*100:.3f}')
print(f'# decayed indices with similarity less than {threshold}: {np.sum(np.array(similarities) < threshold)}')
# %%
decayed_of_interest = np.where(np.array(similarities) < threshold)[0]
decayed_of_interest = [diclist[x]['decayed'] for x in decayed_of_interest]
decayed_of_interest = np.array(decayed_of_interest)
# %%
print(get_top_n_indices(dot_products[1000],4)[1:])
print(cluster_assignment[1000])

""" for i in tqdm(range(num_steps)):
    temp = cc_embeddings[decayed_of_interest[i*step:(i+1)*step]] @ cc_embeddings[decayed_array==0].T
    for j in tqdm(range(len(temp))):
        decayed_ind = decayed_of_interest[i*step+j]
        similar_inds = get_top_n_indices(temp[j], 10)
        diclist_real[decayed_dict[decayed_ind]]['real_similar_exist'] = caption_exist[similar_inds].tolist()
        diclist_real[decayed_dict[decayed_ind]]['real_similar_scores'] = temp[j][similar_inds].tolist() """

# %%

existing = decayed_array==0
cluster_assignments = []
for i in range(cluster_counts):
    cluster_assignments.append(cluster_assignment == i)

# %%

decayed_of_interest_clusters = get_top_n_indices(dot_products[decayed_of_interest[0]],4)[1:]
temp = cc_embeddings[decayed_of_interest[0]] @ cc_embeddings[existing & 
                                                        (cluster_assignments[decayed_of_interest_clusters[0]] |
                                                        cluster_assignments[decayed_of_interest_clusters[1]] |
                                                        cluster_assignments[decayed_of_interest_clusters[2]])].T
similar_inds = get_top_n_indices(temp, 10)
similar_scores = temp[similar_inds]
# %%
similar_scores = np.concatenate(np.array(diclist[decayed_dict[decayed_of_interest[0]]]['similar_scores']), similar_scores)
similar_inds = np.concatenate(np.array(diclist[decayed_dict[decayed_of_interest[0]]]['similar_exist']), similar_inds)
# %%
temp_inds = get_top_n_indices(similar_scores, 10)
similar_scores = similar_scores[temp_inds]
similar_inds = similar_inds[temp_inds]
# %%
np.array(diclist[decayed_dict[decayed_of_interest[0]]]['similar_scores'])

# %%
np.sum((cluster_assignment == decayed_of_interest_clusters[0]) 
 | (cluster_assignment == decayed_of_interest_clusters[1]))
# %%
np.sum(cluster_assignment == decayed_of_interest_clusters[0]) + np.sum(cluster_assignment == decayed_of_interest_clusters[0]) 








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
k = np.linspace(0, 1, 101)
similarities_of_interest_cluster = [diclist[decayed_dict[decayed_of_interest[i]]]['similar_scores'][-1] for i in range(len(decayed_of_interest))]
dist = [np.sum(np.array(similarities_of_interest_cluster) < i) for i in k]
plt.plot(k, dist)
plt.xlabel("Similarity score of 10th most similar in cluster")
plt.ylabel("#")
plt.title("Distribution of similarity scores")
plt.show()

# %%
k = np.linspace(0, 1, 101)
similarities_of_interest_cluster = [diclist_close_3[i]['similar_scores_close_3'][-1] for i in range(len(decayed_of_interest))]
dist = [np.sum(np.array(similarities_of_interest_cluster) < i) for i in k]
plt.plot(k, dist)
plt.xlabel("Similarity score of 10th most similar in cluster")
plt.ylabel("#")
plt.title("Distribution of similarity scores")
plt.show()

# %%
good_th = 0.5
good_ones = np.where(np.array(similarities_of_interest_cluster) < good_th)[0]
good_ones = [diclist_close_3[i]['decayed'] for i in good_ones]
print(len(good_ones))
captions[good_ones]


# %%
print(*captions[good_ones], sep="\n")
# %%
good_caps, good_imgs = [np.where(cc_vs_imagenet[good_ones] > 0.5)][0]
# %%
maxes = np.max(cc_vs_imagenet[good_ones], axis=1)
# %%
k = np.linspace(0, 1, 101)
dist = [np.sum(np.array(maxes) < i) for i in k]
plt.plot(k, dist)
plt.xlabel("Similarity score of 10th most similar in cluster")
plt.ylabel("#")
plt.title("Distribution of similarity scores")
plt.show()
# %%
np.sum(maxes > 0.5)
# %%
np.where(maxes > 0.5)[0]
# %%
caps = [captions[good_ones[i]] for i in good_caps]
labs = [imagenet_classes_short[i] for i in good_imgs]
for i in range(len(caps)):
    print(f'label: {labs[i]}, \t caption: {caps[i]}\n')
# %%
# bunlari imagenet21k ile de karsilastir
# sadece existing sample lara uzak olmasi degil baska decayedlere yakin olmasina da dikkat et
# clusteringde yaparken her bir sample illa bir imagenete atanmisti, bunu belli bir threshold ustunde hepsine ata
# %%

# %%

# %%

# %%

# %%





















# TODO: asil clusteri disindakilere cok daha yakin, neden boyle olduguna bak. sikinti olabilir
# bu 3 en yakin ile olani true ile karsilastir

# %%
with open('diclist_real.json', 'r') as fin:
    diclist_real = json.load(fin)

# %%
count_3 = 0
count_1 = 0
for i in range(len(decayed_of_interest)):
    if diclist_close_3[i]['similar_exist_close_3'][-1] == diclist_real[i]['real_similar_exist'][-1]:
        count_3 += 1
    decayed_ind = decayed_of_interest[i]
    if diclist[decayed_dict[decayed_ind]]['similar_exist'][-1] == diclist_real[i]['real_similar_exist'][-1]:
        count_1 += 1

print(count_1)
print(count_3)

# %%













# %%
diclist_close_3[0]
# %%
diclist[decayed_dict[decayed_of_interest[0]]]
    

# %%
a = np.where(decayed_of_interest == decayed_ind)

# %%
np.where(clusters_decayed_of_interest[0] == decayed_ind)
# %%





# %%

for i in tqdm(range(len(decayed_of_interest))):
    decayed_ind = decayed_of_interest[i]
    decayed_of_interest_clusters = get_top_n_indices(dot_products[decayed_ind],4)[1:]
    temp = cc_embeddings[decayed_ind] @ cc_embeddings[existing & 
                                                        (cluster_assignments[decayed_of_interest_clusters[0]] |
                                                        cluster_assignments[decayed_of_interest_clusters[1]] |
                                                        cluster_assignments[decayed_of_interest_clusters[2]])].T
    similar_inds = get_top_n_indices(temp, 10)
    similar_scores = temp[similar_inds]
    similar_scores = np.concatenate((np.array(diclist[decayed_dict[decayed_ind]]['similar_scores']), similar_scores))
    temp_inds = get_top_n_indices(similar_scores, 10)
    similar_scores = similar_scores[temp_inds]
    diclist_close_3[i]['similar_scores_close_3'] = similar_scores.tolist()


# %%
np.concatenate((np.array(diclist[decayed_dict[decayed_ind]]['similar_scores']), similar_scores))
# %%
np.array(diclist[decayed_dict[decayed_ind]]['similar_scores'])


# %%

# %%

# %%

# %%

























































































# %%
relevant_labels, relevant_clusters = find_matching_labels_and_clusters(cluster_assignment,
                                         imagenet_assignment, decayed_indices,
                                         imagenet_classes_short, imagenet_classes_long,
                                         imagenet_element_count_threshold  = 1000,
                                         imagenet_percentage_in_cluster_threshold  = 0.3,
                                         cluster_percentage_in_imagenet_threshold  = 0.3,
                                         decay_percentage_of_label_in_cluster_threshold = 0.15)
# %%
txt = 6
plt.hist(decayed_txt_vs_embeddings[txt], bins=np.linspace(0,1,100),alpha=0.6)
plt.hist(existing_txt_vs_embeddings[txt], bins=np.linspace(0,1,100),alpha=0.6)
plt.show()

# %%
txt = 4
plt.hist(decayed_txt_vs_embeddings[txt], bins=100)
plt.show()
plt.hist(decayed_of_interest_txt_vs_embeddings[txt], bins=100)
plt.show()
plt.hist(existing_txt_vs_embeddings[txt], bins=100)
plt.show()
plt.hist(existing_txt_vs_embeddings[txt,existing_txt_vs_embeddings[txt]>0.4], bins=100)
plt.show()
# %%
txt = 4
plt.hist(decayed_of_interest_txt_vs_embeddings[txt,decayed_of_interest_txt_vs_embeddings[txt] > 0.4],
        bins=10,alpha=0.45)
#plt.show()
plt.hist(existing_txt_vs_embeddings[txt,existing_txt_vs_embeddings[txt]>0.4], bins=100,alpha=0.45)
plt.show()
# %%
txt_vs_embeddings[3][decayed_of_interest]
# %%
x = np.where((txt_vs_embeddings[0] > 0.75) &  (decayed_array==0))[0]
# %%
len(x)
# %%
captions[x]
# %%
diclist[decayed_dict[decayed_of_interest[4]]]
# %%
k = 10
l = 2
print(captions[decayed_of_interest[k]])
print(captions[diclist[decayed_dict[decayed_of_interest[k]]]['similar_exist'][l]])
print(url[decayed_of_interest[k]])
print(url[diclist[decayed_dict[decayed_of_interest[k]]]['similar_exist'][l]])
# %%
diclist[decayed_dict[decayed_of_interest[k]]]
# %%
# decayin sadece digerlerinden uzak olmasi degil ayni zamanda en yakin olduklarina cok da yakin olmamasi lazim
# bu 04den buyuk existinglerin neden diclist similar da olmadigina bak (compare similarities)
# %%
cc_embeddings[decayed_of_interest[k]] @ cc_embeddings[diclist[decayed_dict[decayed_of_interest[k]]]['similar_exist'][l]]
# %%
cc_embeddings[decayed_of_interest[k]] @ cc_embeddings[x].T
# %%
cluster_assignment[decayed_of_interest[k]]
# %%
cluster_assignment[x]
# %%
np.linspace(0,10,11)
# %%
x = np.where((txt_vs_embeddings[6] > 0.75) &  (decayed_array==0))[0]
print(len(x))
print(captions[x])
# %%
step = 5000
num_steps = len(decayed_of_interest)//step + 1

diclist_real = [{'decayed':x,
            'real_similar_exist':None,
            'real_similar_scores':None} for x in decayed_of_interest]

decayed_interest_dict = {}
for i in range(len(decayed_of_interest)):
    decayed_interest_dict[decayed_of_interest[i]] = i

caption_exist = np.where(decayed_array==0)[0]

""" for i in tqdm(range(num_steps)):
    temp = cc_embeddings[decayed_of_interest[i*step:(i+1)*step]] @ cc_embeddings[decayed_array==0].T
    for j in tqdm(range(len(temp))):
        decayed_ind = decayed_of_interest[i*step+j]
        similar_inds = get_top_n_indices(temp[j], 10)
        diclist_real[decayed_dict[decayed_ind]]['real_similar_exist'] = caption_exist[similar_inds].tolist()
        diclist_real[decayed_dict[decayed_ind]]['real_similar_scores'] = temp[j][similar_inds].tolist() """

# %%

temp = cc_embeddings[decayed_of_interest] @ cc_embeddings[decayed_array==0].T

# %%
for j in tqdm(range(len(temp))):
    decayed_ind = decayed_of_interest[j]
    similar_inds = get_top_n_indices(temp[j], 10)
    diclist_real[decayed_interest_dict[decayed_ind]]['real_similar_exist'] = caption_exist[similar_inds].tolist()
    diclist_real[decayed_interest_dict[decayed_ind]]['real_similar_scores'] = temp[j][similar_inds].tolist()
 
# %%

similarities_of_interest_real = [diclist_real[i]['real_similar_scores'][-1] for i in range(len(diclist_real))]
plt.hist(similarities_of_interest_real, bins=100)
plt.xlabel("Similarity score of 10th most similar in cluster")
plt.ylabel("Count")
plt.show()
# %%
k = np.linspace(0, 1, 100)
dist = [np.sum(np.array(similarities_of_interest_real) < i) for i in k]
plt.plot(k, dist)
plt.xlabel("Similarity score of 10th most similar in cluster")
plt.ylabel("#")
plt.title("Distribution of similarity scores")
plt.show()
# %%
similarities_of_interest_cluster = [diclist[decayed_dict[decayed_of_interest[i]]]['similar_scores'][-1] for i in range(len(decayed_of_interest))]
dist = [np.sum(np.array(similarities_of_interest_cluster) < i) for i in k]
plt.plot(k, dist)
plt.xlabel("Similarity score of 10th most similar in cluster")
plt.ylabel("#")
plt.title("Distribution of similarity scores")
plt.show()
# %%
diclist[decayed_dict[decayed_of_interest[0]]]

# %%
threshold = 0.4
decayed_of_interest_real = np.where(np.array(similarities_of_interest_real) < threshold)[0]
decayed_of_interest_real = np.array([diclist_real[x]['decayed'] for x in decayed_of_interest_real])
# %%
captions[decayed_of_interest_real]
# %%
""" with open('diclist_real.json', 'w') as fout:
    json.dump(diclist_real, fout) """
with open('diclist_real.json', 'r') as fin:
    diclist_real = json.load(fin)
# %%
diclist_real[0]['decayed']
# %%
for i in range(len(diclist_real)):
    diclist_real[i]['decayed'] = int(diclist_real[i]['decayed'])
# %%
