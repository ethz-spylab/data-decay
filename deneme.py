# %%
import clip
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm
# %%
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
device = "cuda:7"
model.to(device)
# %%
df_train = pd.read_csv("/data/projects/data-decay/cc3m/cc3m_2023/Train_GCC-training.tsv", 
                       sep='\t', names=["caption","url"], usecols=range(0,2))
# %%
step_size = 1000
caption_size = len(df_train)
required_steps = math.ceil(caption_size / step_size)
# %%
df = []
for i in tqdm(range(required_steps)):
    torch.cuda.empty_cache()
    texts = [df_train['caption'][x] for x in range(i*step_size,(i+1)*step_size)]
    inputs = processor(text=texts, return_tensors="pt", padding=True,truncation=True).to(device)
    with torch.no_grad():
        outputs = model.text_model(**inputs)
        txt_embeds = outputs[1]
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
    df.append(txt_embeds.cpu().numpy())
# %%
df
# %%
# merge the list of arrays into a single array
df = np.concatenate(df, axis=0)
# %%
df
# %%
df.shape
# %%
# create numpy array from 0 to 104
a = np.arange(105)
step_size = 10
required_steps = math.ceil(len(a) / step_size)
# %%
b = []
for i in range(required_steps):
    b.append(a[i*step_size:(i+1)*step_size])
# %%
b
# %%
# load "/data/projects/data-decay/cc3m/cc3m_2023/captions/Train_GCC-training.tsv" to df_train
df_train = pd.read_csv("/data/projects/data-decay/cc3m/cc3m_2023/captions/Train_GCC-training.tsv", 
                       sep='\t', names=["caption","url"], usecols=range(0,2))
# %%
df_train["caption"]
# %%
len(df_train)
# %%
step_size = 1000
caption_size = len(df_train)
required_steps = math.ceil(caption_size / step_size)
# %%
for i in range(5):
    print(i)
# %%
import json
a = json.load(open('/data/projects/data-decay/cc3m/script_tests/results/summary.json'))
# %%
a
# %%
