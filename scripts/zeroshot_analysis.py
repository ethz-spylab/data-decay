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

from utils import sort_list_by_occurences, get_diff_percent, get_precision_recall_topk
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
precision_11_top5, recall_11_top5 = get_precision_recall_topk(run11_logits, targets, topk=5)
precision_10_top5, recall_10_top5 = get_precision_recall_topk(run10_logits, targets, topk=5)

# Switch the nan values to 0s
precision_11_top5 = np.nan_to_num(precision_11_top5, nan=0)
precision_10_top5 = np.nan_to_num(precision_10_top5, nan=0)

plt.plot(precision_11_top5, precision_10_top5, 'p')
plt.xlabel('run_11 precision')
plt.ylabel('run_10 precision')
plt.title('Top-5 precision')
plt.show()

plt.plot(recall_11_top5, recall_10_top5, 'p')
plt.xlabel('run_11 recall')
plt.ylabel('run_10 recall')
plt.title('Top-5 recall')
plt.show()

print(f'run_11 and run_10 top-5 precision corrcoef: {np.corrcoef(precision_11_top5, precision_10_top5)[0, 1]:.4f}, \
    \nrun_11 and run_10 top-5 recall corrcoef: {np.corrcoef(recall_11_top5, recall_10_top5)[0, 1]:.4f}, \
    \nrun_11 average recall: {np.mean(recall_11_top5):.4f}, \
    \nrun_10 average recall: {np.mean(recall_10_top5):.4f} ')

# %%
# Now do the same for top-1 precision and recall
precision_11_top1, recall_11_top1 = get_precision_recall_topk(run11_logits, targets, topk=1)
precision_10_top1, recall_10_top1 = get_precision_recall_topk(run10_logits, targets, topk=1)

# Switch the nan values to 0s
precision_11_top1 = np.nan_to_num(precision_11_top1, nan=0)
precision_10_top1 = np.nan_to_num(precision_10_top1, nan=0)

plt.plot(precision_11_top1, precision_10_top1, 'p')
plt.xlabel('run_11 precision')
plt.ylabel('run_10 precision')
plt.title('Top-1 precision')
plt.show()

plt.plot(recall_11_top1, recall_10_top1, 'p')
plt.xlabel('run_11 recall')
plt.ylabel('run_10 recall')
plt.title('Top-1 recall')
plt.show()

print(f'run_11 and run_10 top-1 precision corrcoef: {np.corrcoef(precision_11_top1, precision_10_top1)[0, 1]:.4f}, \
    \nrun_11 and run_10 top-1 recall corrcoef: {np.corrcoef(recall_11_top1, recall_10_top1)[0, 1]:.4f}, \
    \nrun_11 average recall: {np.mean(recall_11_top1):.4f}, \
    \nrun_10 average recall: {np.mean(recall_10_top1):.4f} ')
# %%
#Now do it for a given top-k
topk = 10
precision_11_topk, recall_11_topk = get_precision_recall_topk(run11_logits, targets, topk=topk)
precision_10_topk, recall_10_topk = get_precision_recall_topk(run10_logits, targets, topk=topk)

# Switch the nan values to 0s
precision_11_topk = np.nan_to_num(precision_11_topk, nan=0)
precision_10_topk = np.nan_to_num(precision_10_topk, nan=0)

plt.plot(precision_11_topk, precision_10_topk, 'p')
plt.xlabel('run_11 precision')
plt.ylabel('run_10 precision')
plt.title(f'Top-{topk} precision')
plt.show()

plt.plot(recall_11_topk, recall_10_topk, 'p')
plt.xlabel('run_11 recall')
plt.ylabel('run_10 recall')
plt.title(f'Top-{topk} recall')
plt.show()

print(f'run_11 and run_10 top-{topk} precision corrcoef: {np.corrcoef(precision_11_topk, precision_10_topk)[0, 1]:.4f}, \
    \nrun_11 and run_10 top-{topk} recall corrcoef: {np.corrcoef(recall_11_topk, recall_10_topk)[0, 1]:.4f}, \
    \nrun_11 average recall: {np.mean(recall_11_topk):.4f}, \
    \nrun_10 average recall: {np.mean(recall_10_topk):.4f} ')
# %%
precision_11_top5.mean()
# %%
recall_11_top5.mean()
# %%
precision_11_top1.mean()
#%%
open_clip.list_pretrained()
# %%
from tqdm import tqdm

# Preload all logits for zeroshots_open_clip.keys()
logits_dict = {}
for model_key in tqdm(zeroshots_open_clip.keys(), desc='Loading logits'):
    logits_dict[model_key] = torch.from_numpy(np.load(ZERO_SHOT_IMAGENET_RESULTS_VAL_OPEN_CLIP / f'{model_key}.npy'))

# Initialize matrices for recalls and correlations
n_models = len(zeroshots_open_clip)
n_classes = IMAGENET1K_COUNT  # Assuming there are 1000 classes in ImageNet
recall_matrix = np.zeros((n_models, n_classes))
correlation_matrix = np.zeros((n_models, n_models))

# Calculate recalls for all models and fill the recall matrix
model_keys = list(zeroshots_open_clip.keys())
for i, model_key in enumerate(tqdm(model_keys, desc='Calculating recalls')):
    _, recall = get_precision_recall_topk(logits_dict[model_key], targets, topk=5)
    recall = np.nan_to_num(recall, nan=0)  # Convert NaNs to zeros
    recall_matrix[i, :] = recall

# Calculate pairwise correlations and fill the correlation matrix
for i in tqdm(range(n_models), desc='Calculating correlations'):
    for j in range(n_models):
        correlation_matrix[i, j] = np.corrcoef(recall_matrix[i, :], recall_matrix[j, :])[0, 1]

# Save the recall matrix and the correlation matrix into a file
with open(DATA_FOLDER / 'recall_and_correlation_matrices.pkl', 'wb') as f:
    pickle.dump({
        'recall_matrix': recall_matrix,
        'correlation_matrix': correlation_matrix,
        'model_keys': model_keys
    }, f)

# %%
# WHere did you save those?
print(DATA_FOLDER / 'recall_and_correlation_matrices.pkl')
# %%
# Load the recall and correlation matrices in pandas. Display it nicely with seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open(DATA_FOLDER / 'recall_and_correlation_matrices.pkl', 'rb') as f:
    data = pickle.load(f)
recall_matrix = data['recall_matrix']
correlation_matrix = data['correlation_matrix']

# %%
# Recall matrix
recall_matrix_df = pd.DataFrame(recall_matrix, columns=IMAGENET_CLASSNAMES)
recall_matrix_df.index = data['model_keys']
recall_matrix_df

# %%
# Correlation matrix
correlation_matrix_df = pd.DataFrame(correlation_matrix, columns=data['model_keys'])
correlation_matrix_df.index = data['model_keys']
correlation_matrix_df

# %%
# Plot the correlation matrix
plt.figure(figsize=(100, 100))
sns.heatmap(correlation_matrix_df, annot=True, cmap='coolwarm')
plt.show()


# %%
# Find top 100 non-diagonal correlations. Print the corresponding model pairs and their correlations
# %%
# Mask the diagonal by setting it to a low value that cannot be in the top correlations
np.fill_diagonal(correlation_matrix, np.nan)

# Flatten the matrix and sort by correlation, take the top 100
correlations_flat = correlation_matrix.flatten()
top_indices = np.argsort(-correlations_flat)[:100]  # Negative sign for descending sort

# Convert the 1D indices back to 2D indices
top_row_indices, top_col_indices = np.unravel_index(top_indices, correlation_matrix.shape)

# Print the corresponding model pairs and their correlations
for i in range(len(top_row_indices)):
    model1 = data['model_keys'][top_row_indices[i]]
    model2 = data['model_keys'][top_col_indices[i]]
    corr_value = correlation_matrix[top_row_indices[i], top_col_indices[i]]
    print(f"{model1} and {model2} have a correlation of {corr_value:.4f}")

# %%
# print overall recalls for each model
for i, model_key in enumerate(data['model_keys']):
    print(f'{model_key} average recall: {recall_matrix[i, :].mean():.4f}')

# %%
# Do the full same thing with runs in folder val_cc3m
# Preload all logits for zeroshots_cc3m.keys()
# Preload all logits for zeroshots_cc3m.keys()
logits_dict_cc3m = {}
for model_key in tqdm(zeroshots_cc3m.keys(), desc='Loading logits'):
    logits_np = np.load(ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M / f'{model_key}.npy')
    logits_dict_cc3m[model_key] = logits_np

# Initialize matrices for recalls
n_models_cc3m = len(zeroshots_cc3m)
recall_matrix_cc3m = np.zeros((n_models_cc3m, n_classes))

# Calculate recalls for all models and fill the recall matrix
model_keys_cc3m = list(zeroshots_cc3m.keys())
targets_np = targets.numpy()  # Convert targets to NumPy array

for i, model_key in enumerate(tqdm(model_keys_cc3m, desc='Calculating recalls')):
    logits_tensor = torch.from_numpy(logits_dict_cc3m[model_key])
    _, recall = get_precision_recall_topk(logits_tensor, torch.from_numpy(targets_np), topk=5)
    recall = np.nan_to_num(recall.numpy(), nan=0)  # Convert NaNs to zeros
    recall_matrix_cc3m[i, :] = recall

# Calculate pairwise correlations and fill the correlation matrix
correlation_matrix_cc3m = np.zeros((n_models_cc3m, n_models_cc3m))

for i in tqdm(range(n_models_cc3m), desc='Calculating correlations'):
    for j in range(i + 1, n_models_cc3m):  # Only compute upper triangle
        correlation_matrix_cc3m[i, j] = np.corrcoef(recall_matrix_cc3m[i, :], recall_matrix_cc3m[j, :])[0, 1]

# Mirror the upper triangle to the lower triangle
correlation_matrix_cc3m += correlation_matrix_cc3m.T - np.diag(np.diag(correlation_matrix_cc3m))

# Save the recall matrix and the correlation matrix into a file for val_cc3m
with open(ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M / 'recall_and_correlation_matrices_cc3m.pkl', 'wb') as f:
    pickle.dump({
        'recall_matrix': recall_matrix_cc3m,
        'correlation_matrix': correlation_matrix_cc3m,
        'model_keys': model_keys_cc3m
    }, f)

# Mask the diagonal by setting it to NaN
np.fill_diagonal(correlation_matrix_cc3m, np.nan)

# Flatten the matrix and sort by correlation, take the top 100
correlations_flat_cc3m = correlation_matrix_cc3m.flatten()
top_indices_cc3m = np.argsort(-correlations_flat_cc3m)[:100]  # Negative sign for descending sort

# Convert the 1D indices back to 2D indices
top_row_indices_cc3m, top_col_indices_cc3m = np.unravel_index(top_indices_cc3m, correlation_matrix_cc3m.shape)

# Print the corresponding model pairs and their correlations for val_cc3m
for i in range(len(top_row_indices_cc3m)):
    model1 = model_keys_cc3m[top_row_indices_cc3m[i]]
    model2 = model_keys_cc3m[top_col_indices_cc3m[i]]
    corr_value = correlation_matrix_cc3m[top_row_indices_cc3m[i], top_col_indices_cc3m[i]]
    print(f"{model1} and {model2} have a correlation of {corr_value:.4f}")

# %%
