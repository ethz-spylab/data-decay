# %%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"
RUN10 = Path("/data/cc3m/carlini_logs/logs/run_10/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
RUN11 = Path("/data/cc3m/carlini_logs/logs/run_11/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
RUN52 = Path("/data/cc3m/carlini_logs/logs/run_52/2023_04_10-14_15_11-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_FOLDERS_CLASSES_MAPPING = DATA_FOLDER / "imagenet_folder_label_mapping.txt"
IMAGENET_RUNS = DATA_FOLDER / "imagenet_runs"
RUN10_SAVE_FOLDER = IMAGENET_RUNS / "run10"
RUN11_SAVE_FOLDER = IMAGENET_RUNS / "run11"
import open_clip
import numpy as np
from collections import Counter
import os
from PIL import Image
import glob
import torch
from tqdm import tqdm

from utils import (plot_missing_num_perc, get_relevant_captions_and_urls, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters,get_label_cluster_matching_captions_urls,
    fast_load, fast_save, dot_products_distances)

# %%
device = torch.device("cuda:5")

model_10, _, preprocess_10 = open_clip.create_model_and_transforms(
    'RN50', 
    pretrained=str(RUN10), device=device)

model_11, _, preprocess_11 = open_clip.create_model_and_transforms(
    'RN50', 
    pretrained=str(RUN52), device=device)

tokenizer = open_clip.get_tokenizer('RN50')
# %%
imagenet_classes_short = []
with open(IMAGENET_CLASSES_SHORT, "r") as f:
    for line in f:
        imagenet_classes_short.append(line.strip())

imagenet_classes_long = []
with open(IMAGENET_CLASSES_LONG, "r") as f:
    for line in f:
        imagenet_classes_long.append(line.strip())

imagenet_folders = []
imagenet_classes_long_updated = []
with open(IMAGENET_FOLDERS_CLASSES_MAPPING, "r") as f:
    for line in f:
        imagenet_folders.append(line.strip().split(" ")[0])
        imagenet_classes_long_updated.append(line.strip().split(" ")[1].replace("_"," "))

# %%
imagenet_classes_prefix = ["This is a " + label for label in imagenet_classes_long_updated]
# %%
save_counter = 0
text_prep = tokenizer(imagenet_classes_prefix).to(device)
for i in tqdm(range(len(imagenet_folders))):
    image_jpgs = os.listdir(IMAGENET_TRAIN_FOLDER / imagenet_folders[i])
    images = [Image.open(IMAGENET_TRAIN_FOLDER / imagenet_folders[i] / l) for l in image_jpgs]

    image_prep_10 = [preprocess_10(image).unsqueeze(0) for image in images]
    image_prep_11 = [preprocess_11(image).unsqueeze(0) for image in images]

    image_prep_10 = torch.tensor(np.vstack(image_prep_10)).to(device)
    image_prep_11 = torch.tensor(np.vstack(image_prep_11)).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model_10.encode_image(image_prep_10)
        text_features = model_10.encode_text(text_prep)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    comparison_10 = image_features @ text_features.T
    comparison_10 = comparison_10.cpu().numpy().astype('float16')

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model_11.encode_image(image_prep_11)
        text_features = model_11.encode_text(text_prep)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    comparison_11 = image_features @ text_features.T
    comparison_11 = comparison_11.cpu().numpy().astype('float16')

    if i == 0:
        comparison_10_all = comparison_10.copy()
        comparison_11_all = comparison_11.copy()
    elif i == len(imagenet_folders) - 1:
        comparison_10_all = np.vstack((comparison_10_all, comparison_10))
        comparison_11_all = np.vstack((comparison_11_all, comparison_11))
        address_10 = RUN10_SAVE_FOLDER / f"comparison_10_{save_counter}.npy"
        address_11 = RUN11_SAVE_FOLDER / f"comparison_11_{save_counter}.npy"
        fast_save(str(address_10), comparison_10_all)
        fast_save(str(address_11), comparison_11_all)
    elif (i+1) % 200 == 0:
        address_10 = RUN10_SAVE_FOLDER / f"comparison_10_{save_counter}.npy"
        address_11 = RUN11_SAVE_FOLDER / f"comparison_11_{save_counter}.npy"
        fast_save(str(address_10), comparison_10_all)
        fast_save(str(address_11), comparison_11_all)
        comparison_10_all = comparison_10.copy()
        comparison_11_all = comparison_11.copy()
        save_counter += 1
    else:
        comparison_10_all = np.vstack((comparison_10_all, comparison_10))
        comparison_11_all = np.vstack((comparison_11_all, comparison_11))

# %%
for i in tqdm(range(5)):
    if i == 0:
        c10_all = fast_load(str(RUN10_SAVE_FOLDER / f"comparison_10_{i}.npy"))
        c11_all = fast_load(str(RUN11_SAVE_FOLDER / f"comparison_11_{i}.npy"))
    else:
        c10_all = np.vstack((c10_all, fast_load(str(RUN10_SAVE_FOLDER / f"comparison_10_{i}.npy"))))
        c11_all = np.vstack((c11_all, fast_load(str(RUN11_SAVE_FOLDER / f"comparison_11_{i}.npy"))))
# %%
ground_truth = []
for i in range(len(imagenet_folders)):
    ground_truth.extend([[i] for _ in range(len(os.listdir(IMAGENET_TRAIN_FOLDER / imagenet_folders[i])))])

ground_truth = np.array(ground_truth)
ground_truth = np.reshape(ground_truth, (-1,))
# %%

c10_all_max = np.argmax(c10_all, axis=1)
c11_all_max = np.argmax(c11_all, axis=1)
# %%
run10_acc_ave = np.sum(c10_all_max == ground_truth)/len(ground_truth)
run11_acc_ave = np.sum(c11_all_max == ground_truth)/len(ground_truth)
print(f"Run 10 accuracy: {run10_acc_ave*100:.2f}%")
print(f"Run 11 accuracy: {run11_acc_ave*100:.2f}%")
# %%
num_of_cases =[]
c10_correct = []
c11_correct = []
for i in range(len(imagenet_folders)):
    num_of_cases.append(len(os.listdir(IMAGENET_TRAIN_FOLDER / imagenet_folders[i])))
    c10_correct.append(np.sum(c10_all_max[sum(num_of_cases[:i]):sum(num_of_cases[:i+1])] == i))
    c11_correct.append(np.sum(c11_all_max[sum(num_of_cases[:i]):sum(num_of_cases[:i+1])] == i))

num_of_cases = np.array(num_of_cases)
c10_correct = np.array(c10_correct)
c11_correct = np.array(c11_correct)

# %%
c10_c_acc = c10_correct / num_of_cases
c11_c_acc = c11_correct / num_of_cases

# %%
k10 = np.argmax(c10_c_acc)
print(c10_c_acc[k10])
print(imagenet_classes_long_updated[k10])
print("\n")

k11 = np.argmax(c11_c_acc)
print(c11_c_acc[k10])
print(imagenet_classes_long_updated[k10])
# %%
ground_truth.shape
# %%
np.sum(c10_all_max == c11_all_max)/len(c10_all_max)
# %%
i = 517
image_names = os.listdir(IMAGENET_TRAIN_FOLDER / imagenet_folders[i])
images = [Image.open(IMAGENET_TRAIN_FOLDER / imagenet_folders[i] / l) for l in image_names]
image_prep = [preprocess_11(image).unsqueeze(0) for image in images]
text_prep = tokenizer(imagenet_classes_prefix).to(device)
im2 = torch.tensor(np.vstack(image_prep)).to(device)
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model_11.encode_image(im2)
    text_features = model_11.encode_text(text_prep)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

comparison = image_features @ text_features.T
# %%
comparison2 = comparison.cpu().numpy()
res = np.argmax(comparison2, axis=1)
# %%
np.sum(res == 517)
# %%
ground_truth
# %%
ground_truth = np.reshape(ground_truth, (-1,))
# %%
np.sum(c11_all_max[1300:2600] == res)
# %%
preprocess_11
# %%
c11_correct[517]
# %%
imagenet_classes_long_updated[517]
# %%
