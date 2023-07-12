# Creates and saves the embeddings of CC3M captions
# Clip model used is ViT-B/32
# Truncate = True
# The numpy datatype is float64 which leads to an unnecessarily large save file.

import pickle
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import random
import clip
import torch
import time
import math
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

def main():
    print("started running")
    df_train = pd.read_csv("/data/cc3m/cc3m_2023/Train_GCC-training.tsv", sep='\t', names=["caption","url"], usecols=range(0,2))
    model, preprocess = clip.load("ViT-B/32")
    step_size = 10000
    caption_size = len(df_train)
    df = np.empty([caption_size,512])
    req_ = math.floor(caption_size / step_size)
    
    for i in tqdm(range(req_)):
        torch.cuda.empty_cache()
        texts = [df_train['caption'][x] for x in range(i*step_size,(i+1)*step_size)]
        text_tokens = clip.tokenize(texts,truncate=True).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        df[i*step_size:(i+1)*step_size] = text_features.cpu().numpy()
        

    torch.cuda.empty_cache()
    texts = [df_train['caption'][x] for x in range(req_*step_size,caption_size)]
    text_tokens = clip.tokenize(texts,truncate=True).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    df[req_*step_size:caption_size] = text_features.cpu().numpy()
    print("start saving")
    np.save("embeddings/text_embeddings_all.npy",df)
    print("completed")


if __name__ == "__main__":
    main()
