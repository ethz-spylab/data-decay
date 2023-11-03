# %%
import re
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
IMAGENET21K_LONG = DATA_FOLDER / "imagenet21k.txt"
IMAGENET21K_SHORT = DATA_FOLDER / "imagenet21k_short.txt"
IMAGENET21K_EMBEDDINGS = EMBEDDINGS_FOLDER / "imagenet21k_embeddings.npy"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
CC_VS_IMAGENET21K_FOLDER = EMBEDDINGS_FOLDER / "CC_vs_imagenet21k"
CC_VS_IMAGENET21K = CC_VS_IMAGENET21K_FOLDER / "CC_vs_imagenet21k.npy"
CC_VS_IMAGENET21K_ASSIGNMENT = EMBEDDINGS_FOLDER / "CC_vs_imagenet21k_assignment.npy"
IMAGENET21K_LENGTH = 21843
from tqdm import tqdm

from utils import (plot_missing_num_perc, get_relevant_captions_and_urls, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters,get_label_cluster_matching_captions_urls,
    fast_load, fast_save)
# %%

""" imagenet_21k = []
with open('imagenet21k_wordnet_lemmas.txt', 'r') as f:
    for line in f:
        imagenet_21k.append(line.strip()) """
# %%
""" for i in range(len(imagenet_21k)):
    imagenet_21k[i] = imagenet_21k[i].replace('_', ' ') """

# %%
""" with open(IMAGENET21K_LONG, 'w') as f:
    for item in imagenet_21k:
        f.write("%s\n" % item) """
# %%

imagenet21k = []
with open(IMAGENET21K_LONG, 'r') as f:
    for line in f:
        imagenet21k.append("This is a " + line.strip())
# %%
""" model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:0"
model.to(device)
step_size = 1000
num_steps = len(imagenet21k) // step_size + 1
for i in tqdm(range(num_steps)):
    inputs = processor(text=imagenet21k[i*step_size:(i+1)*step_size], return_tensors="pt", padding=True,truncation=True).to(device)
    with torch.no_grad():
        outputs = model.text_model(**inputs)
        txt_embeds = outputs[1]
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
    txt_embeds = txt_embeds.cpu().numpy().astype('float32')
    if i == 0:
        imagenet21k_embeddings = txt_embeds
    else:
        imagenet21k_embeddings = np.vstack((imagenet21k_embeddings,txt_embeds)) """
# %%
""" imagenet21k_embeddings.shape """

# %%

""" np.save(IMAGENET21K_EMBEDDINGS, imagenet21k_embeddings) """
# %%
cc = np.load(CC_EMBEDDINGS_FOLDER)
# %%
imagenet21k_embeddings = np.load(IMAGENET21K_EMBEDDINGS)

# %%
step_size = 1000
num_steps = len(imagenet21k_embeddings) // step_size + 1
for i in tqdm(range(num_steps)):
    x = np.matmul(cc,imagenet21k_embeddings[i*step_size:(i+1)*step_size].T).astype('float16')
    adress = CC_VS_IMAGENET21K_FOLDER / f"CC_vs_imagenet21k_{i}.npy"
    fast_save(str(adress), x)

# %%
CC_VS_IMAGENET21K
# %%
print("CC vs imagenet21k")
# %%
for i in range(10):
    x = CC_VS_IMAGENET21K_FOLDER / f"CC_vs_imagenet21k_{i}.npy"
    print(x)
# %%
imagenet21k = []
with open(IMAGENET21K_LONG, 'r') as f:
    for line in f:
        imagenet21k.append(line.strip())
# %%
imagenet21k_short = []
for i in range(len(imagenet21k)):
    imagenet21k_short.append(imagenet21k[i].split(',')[0])
# %%
len(imagenet21k)
# %%
""" with open(IMAGENET21K_SHORT, 'w') as f:
    for item in imagenet21k_short:
        f.write("%s\n" % item) """
# %%
a = fast_load(str(CC_VS_IMAGENET21K_FOLDER / "CC_vs_imagenet21k_0.npy"))
# %%
a
# %%
a
# %%
cc
# %%
cc = np.load(CC_EMBEDDINGS_FOLDER)
imagenet21k_embeddings = np.load(IMAGENET21K_EMBEDDINGS)

# %%
b = np.matmul(cc,imagenet21k_embeddings[i*step_size:(i+1)*step_size].T)
# %%
a[123,534]
# %%
b[123,534]
# %%
b.type()
# %%
b
# %%
