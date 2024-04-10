# Creates and saves the embeddings of CC3M captions
# Clip model used is openai/clip-vit-large-patch14
# Truncate = True

import numpy as np
import pandas as pd
import torch
import math
from tqdm import tqdm
import argparse
import os

from transformers import CLIPProcessor, CLIPModel

from utils import load_captions

def main(args):

    print("started running get_embeddings.py")

    if os.path.exists(args.dataset_embeddings_path):
        print(f'File {args.dataset_embeddings_path} already exists. Moving on to the next script.')
        return

    captions = load_captions(args.captions_urls_path)
    model = CLIPModel.from_pretrained(args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    device = args.cuda_device
    model.to(device)
    
    step_size = args.step_size
    caption_size = len(captions)
    df = []
    required_steps = math.ceil(caption_size / step_size)

    for i in tqdm(range(required_steps)):
        torch.cuda.empty_cache()
        if i == required_steps - 1:
            texts = [captions[x] for x in range(i*step_size,caption_size)]
        else:
            texts = [captions[x] for x in range(i*step_size,(i+1)*step_size)]
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.text_model(**inputs)
            txt_embeds = outputs[1]
            txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
        df.append(txt_embeds.cpu().numpy())

    df = np.concatenate(df, axis=0)

    print("start saving embeddings")
    if not os.path.exists(os.path.dirname(args.dataset_embeddings_path)):
        print(f'Creating folder {os.path.dirname(args.dataset_embeddings_path)}')
        os.makedirs(os.path.dirname(args.dataset_embeddings_path))

    np.save(args.dataset_embeddings_path, df)
    print("completed saving embeddings")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", type=bool, default=True, help="Whether to print verbose output")
    p.add_argument("--captions_urls_path", type=str, required=True, help="Location of the captions and urls")
    p.add_argument("--model_name", type=str, required=True, help="Model to use for the embeddings")
    p.add_argument("--cuda_device", type=int, required=True, help="Cuda device to use")
    p.add_argument("--step_size", type=int, required=True, help="Step size for calculating embeddings")
    p.add_argument("--dataset_embeddings_path", type=str, required=True, help="Dataset embedding location")
    args = p.parse_args()
    main(args)