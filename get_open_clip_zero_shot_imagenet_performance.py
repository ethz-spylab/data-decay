# %%
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import open_clip
from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
IMAGENET_FOLDER = Path("/data/imagenet")
IMAGENET_TRAIN_FOLDER = IMAGENET_FOLDER / "train"
IMAGENET_VAL_FOLDER = IMAGENET_FOLDER / "val"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS = DATA_FOLDER / "open_clip_zero_shot_imagenet_results"
OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS_VAL = OPEN_CLIP_ZERO_SHOT_IMAGENET_RESULTS / "val"

IMAGENET1K_COUNT = 1000

from utils import create_dataloader, get_accuracy_logits_targets, get_precision_recall, \
    get_autocast

# %%
architecture_dataset = open_clip.list_pretrained()
architecture_dataset
# %%
architecture_dataset[39]
# %%
device = torch.device("cuda:3")
architecture, pretrain_dataset = architecture_dataset[39]
model, _, preprocess = open_clip.create_model_and_transforms(
    architecture, pretrained=pretrain_dataset, device=device)
tokenizer = open_clip.get_tokenizer(architecture)
model.eval()

# %%
class Args:
    def __init__(self,device):
        self.precision = "float32"
        self.device = device
        self.batch_size = 32
args = Args(device)
# %%
imagenet_val_dataloader = create_dataloader(IMAGENET_VAL_FOLDER)
# %%
autocast = get_autocast(args.precision)
with autocast():
    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=True,
    )
# %%
top1, top5, tot_logits, tot_targets = get_accuracy_logits_targets(
    model, classifier, imagenet_val_dataloader, args)
# %%
tot_logits = torch.cat(tot_logits, dim=0)
tot_preds = tot_logits.topk(1, 1, True, True)[1].view(-1)
tot_targets = [tot_targets[i].reshape(1,1) for i in range(len(tot_targets))]
tot_targets = torch.cat(tot_targets, dim=1).view(-1)
tot_correct = tot_preds.eq(tot_targets)

tot_preds = tot_preds.cpu().numpy()
tot_targets = tot_targets.cpu().numpy()
tot_correct = tot_correct.cpu().numpy()

# %%
precision, recall = get_precision_recall(tot_preds, tot_targets)
# %%
zeroshots = {}
zeroshots[architecture + " " + pretrain_dataset] = {"name": precision, "surname": recall}
zeroshots["lmp"] = {"name": "ali", "surname": "veli"}
# %%
zeroshots.keys()
# %%
zeroshots["asd"].keys()
# %%
import pickle 

with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(zeroshots, f)
        
with open('saved_dictionary.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
# %%
loaded_dict
# %%
architecture
# %%
pretrain_dataset
# %%
architecture + " " + pretrain_dataset
# %%
zeroshots
# %%
