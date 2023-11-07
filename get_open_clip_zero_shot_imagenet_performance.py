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

from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_VAL_FOLDER = IMAGENET_FOLDER / "val"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS = DATA_FOLDER / "open_clip_zero_shot_imagenet_results"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS_VAL = OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS / "val"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS_TRAIN = OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS / "train"

IMAGENET1K_COUNT = 1000

from utils import create_dataloader, get_accuracy_logits_targets, get_precision_recall, \
    get_autocast

# %%
device = torch.device("cuda:3")
class Args:
    def __init__(self,device):
        self.precision = "float32"
        self.device = device
        self.batch_size = 32
args = Args(device)

architecture_dataset = open_clip.list_pretrained()
zeroshots = {}
# %%


if (len(sys.argv) == 1) or (sys.argv[1] == "val"):
    IMAGENET_DATALOADER = IMAGENET_VAL_FOLDER
    SAVE_FOLDER = OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS_VAL
    zeroshots_name = "zeroshots_val.pkl"
elif sys.argv[1] == "train":
    IMAGENET_DATALOADER = IMAGENET_TRAIN_FOLDER
    SAVE_FOLDER = OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS_TRAIN
    zeroshots_name = "zeroshots_train.pkl"
else:
    raise ValueError("Invalid argument! Only 'train' or 'val' is accepted!")

for architecture, pretrain_dataset in tqdm(architecture_dataset):

    model, _, preprocess = open_clip.create_model_and_transforms(
        architecture, pretrained=pretrain_dataset, device=device)
    tokenizer = open_clip.get_tokenizer(architecture)
    model.eval()

    """ if model.visual.image_size != (224, 224) and model.visual.image_size != 224:
        # Some models have different image sizes, just ignore them
        continue """


    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=False,
        )
    
    imagenet_dataloader = create_dataloader(image_folder = IMAGENET_DATALOADER, transform = preprocess)

    top1, top5, tot_logits, tot_targets = get_accuracy_logits_targets(
        model, classifier, imagenet_dataloader, args)
    
    tot_logits = torch.cat(tot_logits, dim=0)
    tot_preds = tot_logits.topk(1, 1, True, True)[1].view(-1)
    tot_targets = [tot_targets[i].reshape(1,1) for i in range(len(tot_targets))]
    tot_targets = torch.cat(tot_targets, dim=1).view(-1)

    tot_logits = tot_logits.cpu().numpy()
    tot_preds = tot_preds.cpu().numpy()
    tot_targets = tot_targets.cpu().numpy()

    np.save(SAVE_FOLDER / (architecture + " " + pretrain_dataset + ".npy"), tot_logits)

    precision, recall = get_precision_recall(tot_preds, tot_targets)
    zeroshots[architecture + " " + pretrain_dataset] = {"accuracy_1": top1,"accuracy_5":top5 ,"precision": precision, "recall": recall}

    #TODO: save zeroshots dictionary to save folder as pickle
    with open(SAVE_FOLDER / zeroshots_name, 'wb') as f:
        pickle.dump(zeroshots, f)

    # Save the tot_targets
    np.save(SAVE_FOLDER / "targets.npy", tot_targets)








