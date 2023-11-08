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
import argparse

from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_VAL_FOLDER = IMAGENET_FOLDER / "val"
ZERO_SHOT_IMAGENET_RESULTS = DATA_FOLDER / "zero_shot_imagenet_results"
ZERO_SHOT_IMAGENET_RESULTS_VAL = ZERO_SHOT_IMAGENET_RESULTS / "val_open_clip"
ZERO_SHOT_IMAGENET_RESULTS_TRAIN = ZERO_SHOT_IMAGENET_RESULTS / "train_open_clip"
ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M = ZERO_SHOT_IMAGENET_RESULTS / "val_cc3m"
ZERO_SHOT_IMAGENET_RESULTS_TRAIN_CC3M = ZERO_SHOT_IMAGENET_RESULTS / "train_cc3m"
CARLINI_LOGS = Path("/data/cc3m/carlini_logs/logs/")
IMAGENET1K_COUNT = 1000

from utils import create_dataloader, get_accuracy_logits_targets, get_precision_recall, \
    get_autocast


# %%
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="val", help="imagenet val or train")
parser.add_argument("--models", type=str, default="cc3m", help="cc3m or open_clip")

sysargs = parser.parse_args()
# %%
runs = os.listdir(CARLINI_LOGS)
runs_list = []
for run in runs:
    run_folder = CARLINI_LOGS / run
    run_files = os.listdir(run_folder)
    for run_file in run_files:
        architecture = run_file.split("model_")[1].split("-lr")[0]
        if architecture == 'RN50x4':
            continue
        run_file_name = run + " " + architecture
        run_file_checkpoint = run_folder / run_file / "checkpoints"
        run_file_checkpoint_files = os.listdir(run_file_checkpoint)
        for run_file_checkpoint_file in run_file_checkpoint_files:
            if run_file_checkpoint_file.endswith(".pt"):
                run_file_checkpoint_file_path = run_file_checkpoint / run_file_checkpoint_file
                runs_list.append((run_file_name, run_file_checkpoint_file_path))
    
architecture_dataset = open_clip.list_pretrained()
# %%
device = torch.device("cuda:3")
class Args:
    def __init__(self,device):
        self.precision = "float32"
        self.device = device
        self.batch_size = 32
args = Args(device)

zeroshots = {}
# %%
def select_model_from_open_clip(architecture, pretrain_dataset, device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        architecture, pretrained=pretrain_dataset, device=device)
    tokenizer = open_clip.get_tokenizer(architecture)
    model.eval()
    return model, preprocess, tokenizer

def select_model_from_cc3m(architecture, checkpoint_path, device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        architecture, pretrained=str(checkpoint_path), device=device)
    tokenizer = open_clip.get_tokenizer(architecture)
    model.eval()
    return model, preprocess, tokenizer

# %%
if sysargs.dataset == "val":
    IMAGENET_DATALOADER = IMAGENET_VAL_FOLDER
    zeroshots_name = "zeroshots_val.pkl"
elif sysargs.dataset == "train":
    IMAGENET_DATALOADER = IMAGENET_TRAIN_FOLDER
    zeroshots_name = "zeroshots_train.pkl"
else:
    raise ValueError("Invalid argument! Only 'train' or 'val' is accepted!")

# %%
if sysargs.models == "cc3m":
    if sysargs.dataset == "val":
        SAVE_FOLDER = ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M
    elif sysargs.dataset == "train":
        SAVE_FOLDER = ZERO_SHOT_IMAGENET_RESULTS_TRAIN_CC3M
    get_model = select_model_from_cc3m
    args_list = runs_list
elif sysargs.models == "open_clip":
    if sysargs.dataset == "val":
        SAVE_FOLDER = ZERO_SHOT_IMAGENET_RESULTS_VAL
    elif sysargs.dataset == "train":
        SAVE_FOLDER = ZERO_SHOT_IMAGENET_RESULTS_TRAIN
    get_model = select_model_from_open_clip
    args_list = architecture_dataset
else:
    raise ValueError("Invalid argument! Only 'cc3m' or 'open_clip' is accepted!")
# %%

print(f"Running on {sysargs.dataset} dataset with {sysargs.models} models")

# %%
for first_arg, second_arg in tqdm(args_list):

    if sysargs.models == "cc3m":
        run_name = first_arg
        run_path = second_arg
        architecture = run_name.split(" ")[1].split("-lr_")[0]
        model, preprocess, tokenizer = get_model(architecture, run_path, device)
    elif sysargs.models == "open_clip":
        architecture = first_arg
        pretrain_dataset = second_arg
        model, preprocess, tokenizer = get_model(architecture, pretrain_dataset, device)

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

    precision, recall = get_precision_recall(tot_preds, tot_targets)

    if sysargs.models == "cc3m":
        np.save(SAVE_FOLDER / (run_name + ".npy"), tot_logits)
        zeroshots[run_name] = {"accuracy_1": top1,"accuracy_5":top5 ,"precision": precision, "recall": recall}
    elif sysargs.models == "open_clip":
        np.save(SAVE_FOLDER / (architecture + " " + pretrain_dataset + ".npy"), tot_logits)
        zeroshots[architecture + " " + pretrain_dataset] = {"accuracy_1": top1,"accuracy_5":top5 ,"precision": precision, "recall": recall}

    #TODO: save zeroshots dictionary to save folder as pickle
    with open(SAVE_FOLDER / zeroshots_name, 'wb') as f:
        pickle.dump(zeroshots, f)

    # Save the tot_targets
    np.save(SAVE_FOLDER / "targets.npy", tot_targets)
# %%





# %%
