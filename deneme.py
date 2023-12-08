# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch
import json

from transformers import CLIPProcessor, CLIPModel
# %%

class Args:
    def __init__(self):
        self.dataset_embeddings_path = "/data/cc3m/cc3m_2023/embeddings/text_embeddings_L14.npy"
        self.clusters_folder = '/data/cc3m/script_tests/clusters/'
        self.similarity_type = 'dot_products'
        self.captions_urls_path = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
        self.decayed_indices_save_folder = '/data/cc3m/script_tests/decayed_indices/'
        self.ratio_of_decayed_samples = 0.05
        self.targeted_decay_ratio_of_decayed_samples = 0.1 
        self.start_decay_over_number_ratio = 2.0
        self.device = '7'
        self.verbose = True
args = Args()
# %%
captions_urls_path = args.captions_urls_path
captions_urls = pd.read_csv(captions_urls_path, sep="\t", header=None)
captions_urls.columns = ["caption", "url"]
captions = np.array(captions_urls["caption"])
url = np.array(captions_urls["url"])

# Load the dataset embeddings
if args.verbose:
    print("Loading dataset embeddings")
dataset_embeddings_path = args.dataset_embeddings_path
dataset_embeddings = np.load(dataset_embeddings_path)

dataset_size = dataset_embeddings.shape[0]
if args.verbose:
    print(f'Number of dataset samples: {dataset_size}')
# %%
save_file = args.decayed_indices_save_folder
save_file_decayed_samples = os.path.join(save_file, 'decayed_indices.txt')
save_file_combined_decayed_samples = os.path.join(save_file, 'combined_decayed_indices.txt')
# %%
with open(save_file_decayed_samples, 'r') as f:
    decayed_samples = json.load(f)
# %%
with open(save_file_combined_decayed_samples, 'r') as f:
    combined_decayed_samples = json.load(f)
# %%
i = 0
decayed_ones = np.array(decayed_samples[i])
decayed_embes = dataset_embeddings[decayed_ones]

sims = decayed_embes @ dataset_embeddings.T
np.fill_diagonal(sims, 0)
# %%
captions[decayed_ones]
# %%
np.sum(sims > 0.8)
# %%
