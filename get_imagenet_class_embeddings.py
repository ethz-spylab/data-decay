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
    with open('/data/cc3m/imagenet_classes.txt') as f:
        mylist = f.read().splitlines() 
    texts = ["This is a " + element for element in mylist]
    torch.cuda.empty_cache()
    model, preprocess = clip.load("ViT-B/32")
    text_tokens = clip.tokenize(texts,truncate=True).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features_array = text_features.cpu().numpy().astype('float16')
    print("start saving")
    np.save("/data/cc3m/cc3m_2023/embeddings/imagenet_class_embeddings.npy",text_features_array)
    print("completed")

    

if __name__ == "__main__":
    main()