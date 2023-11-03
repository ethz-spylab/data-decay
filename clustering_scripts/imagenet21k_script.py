import re
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
IMAGENET21K = DATA_FOLDER / "imagenet21k.txt"
IMAGENET21K_EMBEDDINGS = EMBEDDINGS_FOLDER / "imagenet21k_embeddings.npy"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
CC_VS_IMAGENET21K = EMBEDDINGS_FOLDER / "CC_vs_imagenet21k.npy"
CC_VS_IMAGENET21K_ASSIGNMENT = EMBEDDINGS_FOLDER / "CC_vs_imagenet21k_assignment.npy"
from tqdm import tqdm

from utils import (plot_missing_num_perc, get_relevant_captions_and_urls, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters,get_label_cluster_matching_captions_urls,
    fast_load, fast_save)

def main():
    print("started running")
    device = torch.device("cuda:7")
    cc = torch.from_numpy(np.load(CC_EMBEDDINGS_FOLDER)).cuda().to(device)
    imagenet21k_embeddings = np.load(IMAGENET21K_EMBEDDINGS)
    imagenet21k_embedded = torch.from_numpy(imagenet21k_embeddings).cuda().to(device)
    device = torch.device("cuda:7")
    cc = torch.from_numpy(np.load(CC_EMBEDDINGS_FOLDER)).cuda().to(device)
    imagenet21k_embeddings = np.load(IMAGENET21K_EMBEDDINGS)
    imagenet21k_embedded = torch.from_numpy(imagenet21k_embeddings).cuda().to(device)
    step_size = 1000
    num_steps = len(imagenet21k_embedded) // step_size + 1
    compare = np.zeros((len(cc),len(imagenet21k_embedded)), dtype='float16')
    for i in tqdm(range(num_steps)):
        compare[:,i*step_size:(i+1)*step_size] = torch.matmul(cc,imagenet21k_embedded[i*step_size:(i+1)*step_size].T).cpu().numpy().astype('float16')
    print("start saving")
    fast_save(str(CC_VS_IMAGENET21K), compare)
    print("completed")

if __name__ == "__main__":
    main()