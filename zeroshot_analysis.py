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
ZERO_SHOT_IMAGENET_RESULTS = DATA_FOLDER / "zero_shot_imagenet_results"
ZERO_SHOT_IMAGENET_RESULTS_VAL_OPEN_CLIP = ZERO_SHOT_IMAGENET_RESULTS / "val_open_clip"
ZERO_SHOT_IMAGENET_RESULTS_TRAIN_OPEN_CLIP = ZERO_SHOT_IMAGENET_RESULTS / "train_open_clip"
ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M = ZERO_SHOT_IMAGENET_RESULTS / "val_cc3m"
ZERO_SHOT_IMAGENET_RESULTS_TRAIN_CC3M = ZERO_SHOT_IMAGENET_RESULTS / "train_cc3m"
ZEROSHOT_NAME = "zeroshots_val.pkl"

from utils import sort_list_by_occurences, get_diff_percent
import numpy as np

IMAGENET1K_COUNT = 1000
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
with open(ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M / ZEROSHOT_NAME, 'rb') as f:
    zeroshots_cc3m = pickle.load(f)

with open(ZERO_SHOT_IMAGENET_RESULTS_VAL_OPEN_CLIP / ZEROSHOT_NAME, 'rb') as f:
    zeroshots_open_clip = pickle.load(f)
# %%
for k in zeroshots_cc3m.keys():
    print(f'{k}: {zeroshots_cc3m[k]["accuracy_1"]}')
# %%
for k in zeroshots_open_clip.keys():
    print(f'{k}: {zeroshots_open_clip[k]["accuracy_1"]}')
# %%
# OPEN_CLIP comparisons
oc1 = "ViT-B-32 commonpool_m_text_s128m_b4k"
oc2 = "ViT-B-32 commonpool_m_basic_s128m_b4k"
compare_results(oc1, zeroshots_open_clip[oc1], oc2, zeroshots_open_clip[oc2])
# %%
# CC3M comparisons
n1 = "run_11 RN50"
n2 = "run_10 RN50"
compare_results(n1, zeroshots_cc3m[n1], n2, zeroshots_cc3m[n2])
# %%
precision_11 = zeroshots_cc3m['run_11 RN50']['precision']
precision_10 = zeroshots_cc3m['run_10 RN50']['precision']
recall_11 = zeroshots_cc3m['run_11 RN50']['recall']
recall_10 = zeroshots_cc3m['run_10 RN50']['recall']

precision_oc1 = zeroshots_open_clip[oc1]['precision']
precision_oc2 = zeroshots_open_clip[oc2]['precision']
recall_oc1 = zeroshots_open_clip[oc1]['recall']
recall_oc2 = zeroshots_open_clip[oc2]['recall']

# TODO: change the nans to 0s in precisions
precision_11 = np.nan_to_num(precision_11, nan=0)
precision_10 = np.nan_to_num(precision_10, nan=0)
precision_oc1 = np.nan_to_num(precision_oc1, nan=0)
precision_oc2 = np.nan_to_num(precision_oc2, nan=0)

# %%
plt.plot(precision_11, precision_10, 'p' ,label='RN50')
plt.xlabel('run_11 precision')
plt.ylabel('run_10 precision')
plt.legend()
plt.show()
# %%
plt.plot(precision_11, precision_10, 'p' ,label='RN50')
plt.plot(precision_oc1, precision_oc2, '*', label='ViT-B-32')
plt.xlabel('precision high acc')
plt.ylabel('precision low acc')
plt.legend()
plt.show()

print(f'{n1} top-1 accuracy: {zeroshots_cc3m[n1]["accuracy_1"]}, \
      \n{n2} top-1 accuracy: {zeroshots_cc3m[n2]["accuracy_1"]}, \
      \n{oc1} top-1 accuracy: {zeroshots_open_clip[oc1]["accuracy_1"]}, \
      \n{oc2} top-1 accuracy: {zeroshots_open_clip[oc2]["accuracy_1"]}')
print(f'RN50 top-1 precision corrcoef: {np.corrcoef(precision_11, precision_10)[0, 1]}, \
      \nViT-B-32 top-1 precision corrcoef: {np.corrcoef(precision_oc1, precision_oc2)[0, 1]}')
# %%
# Now do the above for recall
plt.plot(recall_11, recall_10, 'p' ,label='RN50')
plt.xlabel('run_11 recall')
plt.ylabel('run_10 recall')
plt.legend()
plt.show()
# %%
plt.plot(recall_11, recall_10, 'p' ,label='RN50')
plt.plot(recall_oc1, recall_oc2, '*', label='ViT-B-32')
plt.xlabel('recall high acc')
plt.ylabel('recall low acc')
plt.legend()
plt.show()

print(f'{n1} top-1 accuracy: {zeroshots_cc3m[n1]["accuracy_1"]}, \
        \n{n2} top-1 accuracy: {zeroshots_cc3m[n2]["accuracy_1"]}, \
        \n{oc1} top-1 accuracy: {zeroshots_open_clip[oc1]["accuracy_1"]}, \
        \n{oc2} top-1 accuracy: {zeroshots_open_clip[oc2]["accuracy_1"]}')
print(f'RN50 top-1 recall corrcoef: {np.corrcoef(recall_11, recall_10)[0, 1]}, \
        \nViT-B-32 top-1 recall corrcoef: {np.corrcoef(recall_oc1, recall_oc2)[0, 1]}')

# %%
run11_logits = torch.from_numpy(np.load(ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M / 'run_11 RN50.npy'))
run10_logits = torch.from_numpy(np.load(ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M / 'run_10 RN50.npy'))

targets = torch.from_numpy(np.load(ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M / 'targets.npy'))
# %%
run11_pred = run11_logits.topk(5, 1, True, True)[1]
correct11 = run11_pred.eq(targets.view(-1, 1).expand_as(run11_pred))

run10_pred = run10_logits.topk(5, 1, True, True)[1]
correct10 = run10_pred.eq(targets.view(-1, 1).expand_as(run10_pred))
# %%
correct10.shape
# %%
correct10.sum() / correct10.shape[0]
# %%
zeroshots_cc3m[n2]["accuracy_1"]
# %%
correct10[targets == 1].sum() / (targets == 0).sum()
# %%
(run11_pred==0).sum()

# %%
