#%%
import pickle
import numpy as np
import argparse

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

#%%
# Now let's take two rows that have the same caption                                                                                                                                                                 
i=1940906
j=1601006

# Check they have the same caption                                                                                                                                                                                   
rows = open(args.og_tsv).readlines()
print(rows[i], rows[j])

#%%
# Now let's look them up in the index                                                                                                                                                                                
# We have to add one because csv starts 1-based                                                                                                                                                                      
ii = idx.index(i+1)
jj = idx.index(j+1)

# Now confirm the embeddings are identical                                                                                                                                                                           
print(np.sum(np.abs(embeds[ii] - embeds[jj])))

