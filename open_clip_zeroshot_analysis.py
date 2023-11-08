# %%
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import open_clip
from open_clip import build_zero_shot_classifier, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
import sys, os
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_VAL_FOLDER = IMAGENET_FOLDER / "val"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS = DATA_FOLDER / "open_clip_zero_shot_imagenet_results"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS_VAL = OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS / "val"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS_TRAIN = OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS / "train"
SAVE_FOLDER = OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS_VAL
zeroshots_name = "zeroshots_val.pkl"

from utils import sort_list_by_occurences, get_diff_percent

IMAGENET1K_COUNT = 1000
# %%
with open(SAVE_FOLDER / zeroshots_name, 'rb') as f:
    zeroshots = pickle.load(f)
# %%
architectures_and_pretraindatasets = list(zeroshots.keys())
architectures = []
pretraindatasets = []
for architecture_dataset in architectures_and_pretraindatasets:
    architectures.append(architecture_dataset.split(" ")[0])
    pretraindatasets.append(architecture_dataset.split(" ")[1])
# %%
architectures_dict = sort_list_by_occurences(architectures)
pretraindatasets_dict = sort_list_by_occurences(pretraindatasets)
# %%
def compare_results(name_1, zeroshots_dict_1, 
                    name_2, zeroshots_dict_2):
    
    recall_1 = zeroshots_dict_1['recall']
    recall_2 = zeroshots_dict_2['recall']
    recall_diff_abs, recall_diff_percent = get_diff_percent(recall_1, recall_2, abs_diff=True)
    
    plt.hist(recall_diff_percent, bins=100, range=(-1, 1))
    plt.title(f'{name_1} vs {name_2} recall diff ratio')
    plt.ylabel('# of classes')
    plt.xlabel('recall diff ratio')
    plt.show()

    plt.hist(recall_diff_abs, bins=100, range=(-50, 50))
    plt.title(f'{name_1} vs {name_2} recall diff abs')
    plt.ylabel('# of classes')
    plt.xlabel('recall diff abs')
    plt.show()

    print(f'{name_1} top-1 accuracy: {zeroshots_dict_1["accuracy_1"]}, \
\n{name_2} top-1 accuracy: {zeroshots_dict_2["accuracy_1"]}')
# %%
list(zeroshots.keys())
# %%
for k in zeroshots.keys():
    print(f'{k}: {zeroshots[k]["accuracy_1"]}')
# %%
n1 = "ViT-B-16 laion400m_e32"
n2 = "ViT-B-32 laion400m_e32"
compare_results(n1, zeroshots[n1], n2, zeroshots[n2])
# %%
