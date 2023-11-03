# %%
import re
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
CSAIL_PLACES = DATA_FOLDER / "csail_places.txt"
CSAIL_PLACES_EMBEDDINGS = EMBEDDINGS_FOLDER / "csail_places_embeddings.npy"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
CC_VS_CSAIL_PLACES = EMBEDDINGS_FOLDER / "CC_vs_csail_places.npy"
CC_VS_CSAIL_PLACES_ASSIGNMENT = EMBEDDINGS_FOLDER / "CC_vs_csail_places_assignment.npy"


#%%
#TODO: read csail_places_html.txt

csail_html = []
with open('csail_places_html.txt', 'r') as f:
    for line in f:
        csail_html.append(line.strip())
# %%
csail_img_src = []
for i in range(len(csail_html)):
    if i % 2 == 1:
        csail_img_src.append(re.findall(r'"([^"]*)"', csail_html[i])[0].split('/')[1].split('.')[0])

# %%
csail = []
for i in range(len(csail_img_src)):
    temp = ""
    split = csail_img_src[i].split('_')
    for j in range(1,len(split)):
        if j == len(split)-1:
            temp = temp + split[j]
        else:
            temp = temp + split[j] + " " 
    csail.append(temp)
# %%
csail

# %%
#TODO: write csail to csail_places.txt
with open(CSAIL_PLACES, 'w') as f:
    for item in csail:
        f.write("%s\n" % item)

# %%
#TODO: read csail_places.txt, concatenate with "This is a" and get the embeddings
csail_places = []
with open(CSAIL_PLACES, 'r') as f:
    for line in f:
        csail_places.append("This is a " + line.strip())






# %%
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda:0"
model.to(device)
inputs = processor(text=csail_places, return_tensors="pt", padding=True,truncation=True).to(device)
with torch.no_grad():
    outputs = model.text_model(**inputs)
    txt_embeds = outputs[1]
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True) 
txt_embeds = txt_embeds.cpu().numpy().astype('float32')
# %%
np.save(CSAIL_PLACES_EMBEDDINGS, txt_embeds)

# %%
cc = torch.from_numpy(np.load(CC_EMBEDDINGS_FOLDER)).cuda()
# %%
csail_embedded = torch.from_numpy(txt_embeds).cuda()
# %%
compare = torch.matmul(cc,csail_embedded.T).cpu().numpy()
# %%
np.save(CC_VS_CSAIL_PLACES, compare)
# %%
assignment = np.argmax(compare, axis=1)
# %%
assignment.shape
# %%
np.save(CC_VS_CSAIL_PLACES_ASSIGNMENT, assignment)
# %%
