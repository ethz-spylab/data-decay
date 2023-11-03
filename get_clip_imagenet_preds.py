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

from utils import (plot_missing_num_perc, get_relevant_captions_and_urls, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters,get_label_cluster_matching_captions_urls,
    fast_load, fast_save, dot_products_distances)

IMAGENET1K_COUNT = 1000
#%%
print("Goal is to get preds for model 11")
#%%
imagenet_classes_short = []
with open(IMAGENET_CLASSES_SHORT, "r") as f:
    for line in f:
        imagenet_classes_short.append(line.strip())
# %%
import torch
from torchvision import datasets, transforms

def create_dataloader(image_folder, batch_size=32, shuffle=False, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(image_folder, transform=transform)
    torch.manual_seed(42)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

# Create dataloaders for ImageNet train and test datasets
imagenet_train_dataloader = create_dataloader(IMAGENET_TRAIN_FOLDER)
imagenet_val_dataloader = create_dataloader(IMAGENET_VAL_FOLDER)

#%%
# make data dict s.t. data['imagenet-val'].dataloader is the val dataloader
#data = {
#    'imagenet-val': imagenet_test_dataloader,
#    'imagenet-v2': imagenet_train_dataloader,
#}
class DataloaderWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader
data = {
    'imagenet-val': DataloaderWrapper(imagenet_val_dataloader),
}
#%%
import open_clip
device = torch.device("cuda:3")

model_10, _, preprocess_10 = open_clip.create_model_and_transforms(
    'RN50', 
    pretrained=str(RUN10), device=device)

model_11, _, preprocess_11 = open_clip.create_model_and_transforms(
    'RN50', 
    pretrained=str(RUN11), device=device)

model_10.eval()
model_11.eval()
#%%
# fill in the args
class Args:
    def __init__(self):
        self.precision = "amp"
        self.device = device
        self.distributed = False
        self.horovod = False
        self.batch_size = 32
        self.model = "RN50"
args = Args()

tokenizer = open_clip.get_tokenizer('RN50')
# %%
print('Building zero-shot classifier')
autocast = get_autocast(args.precision)
with autocast():
    classifier_10 = build_zero_shot_classifier(
        model_10,
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=True,
    )

    classifier_11 = build_zero_shot_classifier(
        model_11,
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=True,
    )

#%%
def get_preds_targets(model, classifier, dataloader, args, topk =(1,)):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    tot_targets = []
    tot_preds = []
    print("Running...")
    with torch.no_grad():
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            pred = logits.topk(max(topk), 1, True, True)[1].t()
            tot_targets.extend(target)
            tot_preds.append(pred)

    return tot_preds, tot_targets
# %%
print("Getting predictions for run 10")
topk = (1,5)
imagenet_train_dataloader = create_dataloader(IMAGENET_TRAIN_FOLDER)
imagenet_val_dataloader = create_dataloader(IMAGENET_VAL_FOLDER)
preds_11, targets_11 = get_preds_targets(
    model_11, classifier_11, imagenet_train_dataloader, args, topk=topk)

# %%

tot_preds_11 = [preds_11[i] for i in range(len(preds_11))]
tot_preds_11 = torch.cat(tot_preds_11, dim=1)
tot_targets_11 = [targets_11[i].reshape(1,1) for i in range(len(targets_11))]
tot_targets_11 = torch.cat(tot_targets_11, dim=1)
tot_correct_11 = tot_preds_11.eq(tot_targets_11.expand_as(tot_preds_11))
print("For model 11:")
for i in topk:
    print(f"Accuracy@{i}: {tot_correct_11[:i].sum() / len(tot_correct_11[0]):.4f}")

# %%
print("Saving predictions for run 11")
tot_preds_11_numpy = tot_preds_11.cpu().numpy()
np.save(IMAGENET_PREDS_RUN11, tot_preds_11_numpy)