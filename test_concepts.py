# %%
import os
import dotenv
from pathlib import Path
import random
import json
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt

captions_embedding_path = Path("/data/cc3m/cc3m_2023/embeddings/text_embeddings_L14.npy")
decayed_indices_path = Path('/data/cc3m/script_tests/decayed_indices/combined_decayed_indices.txt')
#decayed_indices_path = Path('/data/cc3m/decayed_indices.json')
generated_captions_path = Path('/data/cc3m/script_tests/results/generated_captions.json')
# %%
with open(generated_captions_path, "r") as f:
    results = json.load(f)

print(f"Number of clusters: {len(results)}")
# %%
combined_results = results.copy()
combined_results = [item for sublist in combined_results for item in sublist]
# %%
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = f'cuda:7'
model.to(device)

dataset_embeddings_path = captions_embedding_path
dataset_embeddings = np.load(dataset_embeddings_path)
# %%
text =["This is " + x for x in combined_results]
#text = ["baseball player hits a double during the first"]
inputs = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
with torch.no_grad():
    outputs = model.text_model(**inputs)
    text_embeds = outputs[1]
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

text_embeds = text_embeds.cpu().numpy().astype('float32')

txt_vs_embeddings = text_embeds @ dataset_embeddings.T
# %%

with open(decayed_indices_path, "r") as f:
    decayed_indices = json.load(f)

decayed_array = np.zeros(len(dataset_embeddings))
decayed_array[decayed_indices] = 1

# %%
for i in range(len(text)):

    txt = i
    th = 0.5
    decayed_txt_vs_embeddings = txt_vs_embeddings[txt,decayed_array==1]
    existing_txt_vs_embeddings = txt_vs_embeddings[txt,decayed_array==0]
    plt.hist(decayed_txt_vs_embeddings, bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.hist(existing_txt_vs_embeddings, bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.legend(["Decayed", "Existing"])
    plt.xlabel("Similarity score")
    plt.ylabel("Count")
    plt.title("\""+text[txt]+"\""+" vs captions")
    plt.show()

    dec = np.sum(decayed_txt_vs_embeddings > th)
    ex = np.sum(existing_txt_vs_embeddings > th)
    print(i)
    print(f'# of decayed: {dec}, \
        \n# of total: {ex+dec}, \
        \n% of decayed: {dec/(ex+dec)*100:.2f}')
# %%
