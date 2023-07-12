"""
There is a dataframe with two columns: URLs and captions.
A subset of the rows, given by indices in a separate txt file, is special.
We want to cluster the whole dataset and then find clusters that are dense with special captions.
Our first try is to run clusters using k-means on the CLIP embeddings of the captions.
"""

#%% Imports
import pickle
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import torch
import sklearn.cluster
import threadpoolctl
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from pathlib import Path
from tqdm import tqdm
import transformers
import torch, torch.nn as nn
import pickle
import open_clip
import numpy as np

from open_clip import create_model_and_transforms
from open_clip import load_model, tokenize, embed_text


#%% Utils
def pw_cosine_distance(input_a, input_b):
    assert input_a.shape == input_b.shape
    input_a = input_a / torch.linalg.norm(input_a)
    input_b = input_b / torch.linalg.norm(input_b)
    return input_a @ input_b.t()

#%%
#%% Load the model
#model, _, preprocess = create_model_and_transforms("ViT-L-14", "openai")
MODEL_NAME = "ViT-B-32-quickgelu"
GPU_ID = 0
model, _, preprocess = create_model_and_transforms(MODEL_NAME, pretrained="laion400m_e32")
device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
model.to(device)

#%% Get text embeds function
def get_text_embeds_simple(clip_model, input : list[str]):
    """Get text embeddings for a list of text"""
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            input = open_clip.tokenize(input).to(device)
            embedding = clip_model.encode_text(input)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding

        
def get_text_embeds(clip_model, input : list[str]):
    """Get text embeddings for a list of text. 
    This is a wrapper around get_text_embeds_simple that handles batching."""
    embeds = []
    bs = 1024 * 16
    for i in tqdm(range(0, len(input), bs)):
        embeds.append(get_text_embeds_simple(clip_model, input[i:i+bs]))
    return torch.cat(embeds)

def test_get_text_embeds():
    text = "A dog is running in the grass."
    text2 = "A cat is running in the grass."
    text3 = "A big window is open."
    embeds = get_text_embeds_simple(model, [text, text2, text3])
    # give me l2 norms of embeds
    print(f"Embedding l2 norms: {torch.linalg.norm(embeds, dim=1)}")
    print(f"Embeddings: {embeds.shape}")
    #print(pw_cosine_distance(embeds[0], embeds[1]).item(), pw_cosine_distance(embeds[0], embeds[2]).item(), pw_cosine_distance(embeds[1], embeds[2]).item())
    print(f"Similarity: {pw_cosine_distance(embeds[0], embeds[1]).item():.3f}, {pw_cosine_distance(embeds[0], embeds[2]).item():.3f}, {pw_cosine_distance(embeds[1], embeds[2]).item():.3f}")

test_get_text_embeds()
#%%
# Load the full data, save the k-means clusters
# TODO

