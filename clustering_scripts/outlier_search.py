#%%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
CC_EMBEDDINGS_FOLDER_FAST = EMBEDDINGS_FOLDER / "text_embeddings_L14_fast.npy"
#CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings.npy"
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
DECAYED_INDICES = DATA_FOLDER / "decayed_indices.txt"
CC_DECAYED_SIMILARITY_DICT = DATA_FOLDER / "cc_decayed_similarity_dict.json"
CC_DECAYED_SIMILARITY_DICT_100 = DATA_FOLDER / "cc_decayed_similarity_dict_100.json"

CLUSTER_CENTERS = EMBEDDINGS_FOLDER / "cluster_centers_L14.npy"
DOT_PRODUCTS = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_L14.npy"
DISTANCES = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_distances_L14.npy"

from transformers import CLIPProcessor, CLIPModel

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch

from tqdm import tqdm
import json
# %%
import sys
import numpy as np
import numpy.lib.format
import struct

def fast_save(file, array):
    magic_string=b"\x93NUMPY\x01\x00v\x00"
    header=bytes(("{'descr': '"+array.dtype.descr[0][1]+"', 'fortran_order': False, 'shape': "+str(array.shape)+", }").ljust(127-len(magic_string))+"\n",'utf-8')
    if type(file) == str:
        file=open(file,"wb")
    file.write(magic_string)
    file.write(header)
    file.write(array.tobytes())

def fast_pack(array):
    size=len(array.shape)
    return bytes(array.dtype.byteorder.replace('=','<' if sys.byteorder == 'little' else '>')+array.dtype.kind,'utf-8')+array.dtype.itemsize.to_bytes(1,byteorder='little')+struct.pack(f'<B{size}I',size,*array.shape)+array.tobytes()

def fast_load(file):
    if type(file) == str:
        file=open(file,"rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))

#%%
def get_top_n_indices(array : np.ndarray, n: int = 10):
    """Returns the indices of the top n elements of an array.
    
    Args:
        array (np.array): The array to find the indices of the top n elements.
        n (int): The number of indices to return.
    Returns:
        np.array: The indices of the top n elements of the array."""
    
    return np.argsort(array)[::-1][:n]

# %%
cc_captions = pd.read_csv(CC_CAPTIONS_DF, sep="\t", header=None)
cc_captions.columns = ["caption", "url"]
captions = np.array(cc_captions["caption"])
url = np.array(cc_captions["url"])
# %%
cc_embeddings = fast_load(str(CC_EMBEDDINGS_FOLDER_FAST))
cluster_centers = np.load(CLUSTER_CENTERS)
dot_products = np.load(DOT_PRODUCTS)
cluster_counts = dot_products.shape[1]

# %%
cluster_assignment = np.argmax(dot_products, axis=1)

# %% 
# TODO: read decayed indices to a list
decayed_indices = []
with open(DECAYED_INDICES, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))

decayed_array = np.zeros(len(cluster_assignment))
decayed_array[decayed_indices] = 1
decayed_indices = np.array(decayed_indices)
# %%
existing_indices = np.where(decayed_array == 0)[0]
# %%
existing_embeddings = cc_embeddings[existing_indices]
decayed_embeddings = cc_embeddings[decayed_indices]
# %%
cc_embeddings[existing_indices]
# %%
from sklearn.neighbors import LocalOutlierFactor
# %%
clf = LocalOutlierFactor(n_neighbors=20, novelty=True)
# %%
clf.fit(existing_embeddings)
# %%
