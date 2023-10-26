# %%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"
IMAGENET_FOLDERS_CLASSES_MAPPING = DATA_FOLDER / "imagenet_folder_label_mapping.txt"
IMAGENET_FOLDERS_CLASSES_MAPPING_ORIGINAL = DATA_FOLDER / "imagenet_folder_label_mapping_original.txt"
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
# %%
import numpy as np
from collections import Counter
import os
from PIL import Image
# %%
imagenet_classes_short = []
with open(IMAGENET_CLASSES_SHORT, "r") as f:
    for line in f:
        imagenet_classes_short.append(line.strip())

imagenet_classes_long = []
with open(IMAGENET_CLASSES_LONG, "r") as f:
    for line in f:
        imagenet_classes_long.append(line.strip())
# %%
imagenet_folder_names = []
imagenet_labels_unordered = []
with open (IMAGENET_FOLDERS_CLASSES_MAPPING_ORIGINAL, "r") as f:
    for line in f:
        imagenet_folder_names.append(line.strip().split(" ")[0])
        imagenet_labels_unordered.append(line.strip().split(" ")[2].replace("_"," "))
# %%
imagenet_classes_short[134] = "crane bird"
imagenet_classes_short[517] = "crane machine"
imagenet_classes_long[134] = "crane bird"
imagenet_classes_long[517] = "crane machine"
imagenet_labels_unordered[428] = "crane bird"
imagenet_labels_unordered[544] = "crane machine"
imagenet_classes_short[639] = "maillot, tank suit"
imagenet_labels_unordered[976] = "maillot, tank suit"
imagenet_labels_long_ordered = []
imagenet_folders_ordered = []
for i in range(len(imagenet_classes_short)):
    for j in range(len(imagenet_labels_unordered)):
        if imagenet_classes_short[i] == imagenet_labels_unordered[j]:
            imagenet_labels_long_ordered.append(imagenet_classes_long[i])
            imagenet_folders_ordered.append(imagenet_folder_names[j])
            break
# %%
with open(IMAGENET_FOLDERS_CLASSES_MAPPING, "w") as f:
    for i in range(len(imagenet_folders_ordered)):
        temp = imagenet_labels_long_ordered[i].replace(" ", "_")
        f.write(f"{imagenet_folders_ordered[i]} {temp}\n")
# %%
