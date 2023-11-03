#%%
import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from precision import get_autocast

import torch
import torch.nn.functional as F

import numpy as np
import torch
from torchvision import datasets, transforms

from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"
RUN10 = Path("/data/cc3m/carlini_logs/logs/run_10/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
RUN11 = Path("/data/cc3m/carlini_logs/logs/run_11/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_VAL_FOLDER = IMAGENET_FOLDER / "val"
IMAGENET_FOLDERS_CLASSES_MAPPING = DATA_FOLDER / "imagenet_folder_label_mapping.txt"
IMAGENET_PREDS_FOLDER = DATA_FOLDER / "imagenet_preds"
IMAGENET_PREDS_RUN10 = IMAGENET_PREDS_FOLDER / "imagenet_preds_run10.npy"
IMAGENET_PREDS_RUN11 = IMAGENET_PREDS_FOLDER / "imagenet_preds_run11.npy"

import matplotlib.pyplot as plt
import os

from utils import (plot_missing_num_perc, get_relevant_captions_and_urls, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters,get_label_cluster_matching_captions_urls,
    fast_load, fast_save, dot_products_distances)

IMAGENET1K_COUNT = 1000

#%%
imagenet_classes_short = []
with open(IMAGENET_CLASSES_SHORT, "r") as f:
    for line in f:
        imagenet_classes_short.append(line.strip())

imagenet_folder_names = []
imagenet_labels_unordered = []
with open (IMAGENET_FOLDERS_CLASSES_MAPPING, "r") as f:
    for line in f:
        imagenet_folder_names.append(line.strip().split(" ")[0])
        imagenet_labels_unordered.append(line.strip().split(" ")[1].replace("_"," "))
# %%
tot_preds_10 = np.load(IMAGENET_PREDS_RUN10)
tot_preds_11 = np.load(IMAGENET_PREDS_RUN11)

tot_targets = []
for k in range(len(imagenet_folder_names)):
    num = len(os.listdir(IMAGENET_TRAIN_FOLDER / imagenet_folder_names[k]))
    tot_targets.extend([k for _ in range(num)])
tot_targets = np.array(tot_targets)

# %%
topk = (1,5)
tot_correct_10 = tot_preds_10 == np.tile(tot_targets.reshape(1,-1), (5,1))
tot_correct_11 = tot_preds_11 == np.tile(tot_targets.reshape(1,-1), (5,1))
print("For model 10:")
for i in topk:
    print(f"Accuracy@{i}: {tot_correct_10[:i].sum() / len(tot_correct_10[0]):.4f}")

print("For model 11:")
for i in topk:
    print(f"Accuracy@{i}: {tot_correct_11[:i].sum() / len(tot_correct_11[0]):.4f}")

# %%
k=1
class_correct_10 = np.zeros(IMAGENET1K_COUNT)
class_total_10 = np.zeros(IMAGENET1K_COUNT)
class_correct_11 = np.zeros(IMAGENET1K_COUNT)
class_total_11 = np.zeros(IMAGENET1K_COUNT)
correct_10 = tot_correct_10[:k]
correct_11 = tot_correct_11[:k]
for i in range(IMAGENET1K_COUNT):
    class_correct_10[i] = np.sum(correct_10[:,tot_targets==i])
    class_total_10[i] = np.sum(tot_targets==i)

    class_correct_11[i] = np.sum(correct_11[:,tot_targets==i])
    class_total_11[i] = np.sum(tot_targets==i)
# %%
class_percent_10 = class_correct_10 / class_total_10
class_percent_11 = class_correct_11 / class_total_11

class_percent_diff = class_percent_11 - class_percent_10
# %%
n = 10
ind = get_top_n_indices(class_percent_diff, n)
plt.bar([imagenet_classes_short[i] for i in ind], class_percent_diff[ind])
plt.xticks(rotation = 45)
plt.show()

for i in range(n):
    print(f"{imagenet_classes_short[ind[i]]} , label number:{ind[i]}\
\n11: {class_percent_11[ind[i]]:.4f}, 10: {class_percent_10[ind[i]]:.4f}, diff: {class_percent_diff[ind[i]]:.4f}")

# %%
print(np.sum(class_percent_diff > 0))
print(np.sum(class_percent_diff < 0))
# %%
class_correct_10[546]
# %%
class_correct_11[546]
# %%
imagenet_classes_short[546]
# %%
