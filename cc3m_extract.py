#%%
import pickle
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--index", default="/data/cc3m_embed/embed_cc3m_2018_b/order0.p", help="Path to index file")
parser.add_argument("--embed", default="/data/cc3m_embed/embed_cc3m_2018_b/txt0.p", help="Path to embedding file")
#parser.add_argument("--output", default="/data/cc3m_embed/embed_cc3m_2018_b/txt0.txt", help="Path to output file")
parser.add_argument('--og_tsv', default="/data/cc3m/cc3m_2023/Train_GCC-training.tsv", help="Path to original tsv file")
args = parser.parse_args(args=[]) if hasattr(__builtins__,'__IPYTHON__') else parser.parse_args()

#%%
# First load the index                                                                                                                                                                                               
idx = pickle.load(open(args.index, "rb"))
# and the embeddings file                                                                                                                                                                                            
embeds = pickle.load(open(args.embed, "rb"))

df = pd.read_csv("/data/cc3m/cc3m_2023/Train_GCC-training.tsv", sep='\t', names=["caption","url"], usecols=range(0,2))
df

#%%
# Now let's take two rows that have the same caption                                                                                                                                                                 
i=1940905
j=1601005

# Check they have the same caption                                                                                                                                                                                   
print(df['caption'][i])
print(df['caption'][j])

#%%
# Now let's look them up in the index                                                                                                                                                                                
# We have to add 2 because some offsets
ii = idx.index(i+2)
jj = idx.index(j+2)

# Now confirm the embeddings are identical, using l_2 norm
print("L2 distance between embeddings: ", np.linalg.norm(embeds[ii] - embeds[jj]))
print("L2 norms of embeddings: ", np.linalg.norm(embeds[ii]), np.linalg.norm(embeds[jj]))


# %%
