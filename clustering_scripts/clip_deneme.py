# %%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_CLASSES_SHORT = DATA_FOLDER / "imagenet_classes.txt"
IMAGENET_CLASSES_LONG = DATA_FOLDER / "imagenet_classes_long.txt"
RUN10 = Path("/data/cc3m/carlini_logs/logs/run_10/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
RUN11 = Path("/data/cc3m/carlini_logs/logs/run_11/2023_03_31-09_33_50-model_RN50-lr_5e-05-b_128-j_8-p_amp/checkpoints") / "epoch_32.pt"
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_FOLDERS_CLASSES_MAPPING = DATA_FOLDER / "imagenet_folder_label_mapping.txt"
import open_clip
# %%
import numpy as np
from collections import Counter
import os
from PIL import Image
import glob
import torch
# %%
device = torch.device("cuda:5")

model_10, _, preprocess_10 = open_clip.create_model_and_transforms(
    'RN50', 
    pretrained=str(RUN10), device=device)

model_11, _, preprocess_11 = open_clip.create_model_and_transforms(
    'RN50', 
    pretrained=str(RUN11), device=device)

tokenizer = open_clip.get_tokenizer('RN50')
# %%
#open_clip.list_pretrained()

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
for i in range(len(imagenet_classes_long)):
    if imagenet_classes_long[i] != imagenet_classes_long_updated[i]:
        print(i)
        print(imagenet_classes_long[i])
        print(imagenet_classes_long_updated[i])
        print("\n")
# %%
i = 517
print(imagenet_classes_long_updated[i])
l = os.listdir(IMAGENET_TRAIN_FOLDER / imagenet_folders[i])
Image.open(IMAGENET_TRAIN_FOLDER / imagenet_folders[i] / l[0])

# %%
i = 517
image_names = os.listdir(IMAGENET_TRAIN_FOLDER / imagenet_folders[i])
images = [Image.open(IMAGENET_TRAIN_FOLDER / imagenet_folders[i] / l) for l in image_names]
# %%
images[756]
# %%
len(images)
# %%
image_prep = [preprocess_11(image).unsqueeze(0) for image in images]
text_prep = tokenizer(imagenet_classes_long_updated).to(device)
# %%
im2 = torch.tensor(np.vstack(image_prep)).to(device)
# %%
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model_11.encode_image(im2)
    text_features = model_11.encode_text(text_prep)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

comparison = image_features @ text_features.T
# %%
comparison2 = comparison.cpu().numpy()

# %%
comparison2
# %%
res = np.argmax(comparison2, axis=1)
# %%
res.shape
# %%
np.sum(res == 517)
# %%
c = Counter(res)
# %%
c.most_common(10)
# %%
imagenet_classes_long_updated[517]
# %%









# %%
import clip
# %%
device = torch.device("cuda:5")
model, preprocess = clip.load("ViT-L/14", device=device)

# %%
images_prep = [preprocess(image)for image in images]
image_input = torch.tensor(np.stack(images_prep)).to(device)
text_tokens = clip.tokenize(["This is " + desc for desc in imagenet_classes_long_updated]).to(device)
# %%
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
# %%
sim = image_features @ text_features.T

# %%
res = np.argmax(sim.cpu().numpy(), axis=1)

# %%
res.shape
# %%
np.sum(res == 517)
# %%
imagenet_classes_long_updated[517]
# %%
images[0]
# %%
k = np.array([[1,2],[5,4],[6,8]])
# %%
np.argmax(k, axis=1)
# %%
comparison3 = comparison.cpu().numpy()
# %%
c4 = np.vstack([comparison2, comparison3])
# %%
c4.shape
# %%
