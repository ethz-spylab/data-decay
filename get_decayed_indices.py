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
        #self.ratio_of_random_decayed_samples = 0.025
        self.ratio_of_random_decayed_samples = 0.1
        self.ratio_of_targeted_decayed_samples = 0.005
        #self.sample = False
        self.sample = True
        self.sample_ratio = 2.0
        self.device = '7'
        self.verbose = True
args = Args()
# %%
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = f'cuda:{args.device}'
model.to(device)
print(f'Got the model to {device}')
# %%
captions_urls_path = args.captions_urls_path
captions_urls = pd.read_csv(captions_urls_path, sep="\t", header=None)
captions_urls.columns = ["caption", "url"]
captions = np.array(captions_urls["caption"])
url = np.array(captions_urls["url"])

# %%
# Load the dataset embeddings
if args.verbose:
    print("Loading dataset embeddings")
dataset_embeddings_path = args.dataset_embeddings_path
dataset_embeddings = np.load(dataset_embeddings_path)

dataset_size = dataset_embeddings.shape[0]
if args.verbose:
    print(f'Number of dataset samples: {dataset_size}')
# %%
text = ["This is a basketball player",
        "This is a laptop",
        "This is a dog",
        "This is a musician playing guitar",
        "This is a hotel lobby",
        "This is a child riding a bike",
        "This is a skyscraper",
        #"This is a person jogging",
        #"This is a birthday cake",
        #"This is a birthday party",
        "This is a living room",
        #"This is a bedroom",
        #"This is a kitchen",
        "This is a bathroom",
        #"This is a lion",
        #"This is a tiger",
        "This is a zebra",
        "This is a birthday cake"]
inputs = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
with torch.no_grad():
    outputs = model.text_model(**inputs)
    txt_embeds = outputs[1]
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
txt_embeds = txt_embeds.cpu().numpy().astype('float32')

txt_vs_embeddings = txt_embeds @ dataset_embeddings.T

targeted_decay_size = round(dataset_size * args.ratio_of_targeted_decayed_samples / len(text))
print(f'Targeted decay size: {targeted_decay_size}')
# %%
for i in range(len(text)):

    txt = i
    th = 0.5
    plt.hist(txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1))
    plt.xlabel("Similarity score")
    plt.ylabel("Count")
    plt.title("\""+text[txt]+"\""+" vs captions")
    plt.show()

    tot = np.sum(txt_vs_embeddings[txt] > th)
    print(i)
    print(f'# of samples with similarity score > {th}: {tot}')
# %%
sample_ratio = args.sample_ratio
if args.sample == False:
    sample_ratio = 1.0
start_decay_size = round(targeted_decay_size * sample_ratio)

thresholds = []
for i in range(len(text)):
    txt_vs_embedding = txt_vs_embeddings[i]

    starting_th = 0.5
    th = starting_th
    th_step = 0.01
    num_over_th = np.sum(txt_vs_embedding > starting_th)

    if num_over_th > start_decay_size:
        while num_over_th > start_decay_size:
            th += th_step
            num_over_th = np.sum(txt_vs_embedding > th)
        th -= th_step
        num_over_th = np.sum(txt_vs_embedding > th)
    else:
        while num_over_th < start_decay_size:
            th -= th_step
            num_over_th = np.sum(txt_vs_embedding > th)
        """ th += th_step
        num_over_th = np.sum(txt_vs_embedding > th) """

    thresholds.append(round(th,2))
# %%
thresholds
# %%
for i in range(len(text)):

    txt = i
    th = thresholds[i]
    plt.hist(txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1))
    plt.xlabel("Similarity score")
    plt.ylabel("Count")
    plt.title("\""+text[txt]+"\""+" vs captions")
    plt.show()

    tot = np.sum(txt_vs_embeddings[txt] > th)
    print(i)
    print(f'# of samples with similarity score > {th}: {tot}')
# %%
decayed_samples = []
for i in range(len(text)):
    txt_vs_embedding = txt_vs_embeddings[i]
    th = thresholds[i]
    if args.sample:
        decayed_samples.append(np.random.choice(np.where(txt_vs_embedding > th)[0],
                                                 size=targeted_decay_size, replace=False).tolist())
    else:
        over_th_decayed_samples = np.where(txt_vs_embedding > th)[0]
        ind = np.argpartition(txt_vs_embedding[over_th_decayed_samples], -targeted_decay_size)[-targeted_decay_size:]
        decayed_samples.append(over_th_decayed_samples[ind].tolist())
# %%
targeted_samples = np.concatenate(decayed_samples)
# %%
remaining_samples = np.setdiff1d(np.arange(dataset_size), targeted_samples)
# %%
np.random.seed(42)
random_decay_size = round(dataset_size * args.ratio_of_random_decayed_samples)
random_decay_samples = np.random.choice(remaining_samples, size=random_decay_size, replace=False)
decayed_samples.append(random_decay_samples.tolist())
# %%
save_file = args.decayed_indices_save_folder
save_file_decayed_samples = os.path.join(save_file, 'decayed_indices.txt')
save_file_combined_decayed_samples = os.path.join(save_file, 'combined_decayed_indices.txt')
# %%
with open(save_file_decayed_samples, 'w') as fout:
    json.dump(decayed_samples, fout)
# %%
combined_decayed_samples = np.unique(np.concatenate(decayed_samples)).tolist()
# %%
with open(save_file_combined_decayed_samples, 'w') as fout:
    json.dump(combined_decayed_samples, fout)
# %%
print(f'Number of targeted decayed samples: {len(targeted_samples)}')
print(f'Number of random decayed samples: {len(random_decay_samples)}')
print(f'Number of total decayed samples: {len(combined_decayed_samples)}')
# %%
decayed_array = np.zeros(dataset_size)
decayed_array[np.array(targeted_samples)] = 1

decayed_txt_vs_embeddings = txt_vs_embeddings[:,decayed_array==1]
existing_txt_vs_embeddings = txt_vs_embeddings[:,decayed_array==0]

for i in range(len(text)):

    txt = i
    th = thresholds[i]
    plt.hist(decayed_txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.hist(existing_txt_vs_embeddings[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.legend(["Decayed", "Existing"])
    plt.xlabel("Similarity score")
    plt.ylabel("Count")
    plt.title("\""+text[txt]+"\""+" vs captions")
    plt.show()

    dec = np.sum(decayed_txt_vs_embeddings[txt] > th)
    ex = np.sum(existing_txt_vs_embeddings[txt] > th)
    print(i)
    print(f'# of decayed: {dec}, \
        \n# of total: {ex+dec}, \
        \n% of decayed: {dec/(ex+dec)*100:.2f}')
# %%
captions[decayed_samples[-1][:10]]
# %%
