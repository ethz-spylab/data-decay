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

IMAGENET1K_COUNT = 1000

# %%
with open(ZERO_SHOT_IMAGENET_RESULTS_VAL_CC3M / "zeroshots_val.pkl", 'rb') as f:
    zeroshots = pickle.load(f)
# %%
for key in zeroshots.keys():
    print(f'{key.split(" ")[0]}, top1: {zeroshots[key]["accuracy_1"]}')
# %%
for i in tqdm(range(10)):
    print("hi")
# %%
