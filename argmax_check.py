#%%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
IMAGENET_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "imagenet_class_embeddings_L14.npy"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings_L14.npy"
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
DOT_PRODUCTS = EMBEDDINGS_FOLDER / "CC_vs_imagenet_L14.npy"  # (cc entries, imagenet classes)
IMAGENET_LABELS = DATA_FOLDER / "imagenet_classes.txt"
LONG_IMAGENET_LABELS = DATA_FOLDER / "imagenet_classes_long.txt"

#%%
from urllib.request import urlopen
import pickle
long_labels = pickle.load(urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
long_labels = [long_labels[i] for i in range(1000)]
# save to file, as text
with open(LONG_IMAGENET_LABELS, 'w') as f:
    for item in long_labels:
        f.write("%s\n" % item)



#%%
# Get the argmax for each CC entry.
import numpy as np
dot_products = np.load(DOT_PRODUCTS)
argmax = np.argmax(dot_products, axis=1)
maxes = np.max(dot_products, axis=1)


#%%
# Make a curve of the number of entries that have the max imagenet similarity larger than x
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()
#sns.set_style("whitegrid")
#sns.set_context("paper", font_scale=1.5)
#plt.figure(figsize=(10, 6))
#plt.hist(dot_products.max(axis=1), bins=100)
#plt.xlabel("Max product with imagenet")
#plt.ylabel("Number of CC entries")
#plt.savefig("max_product_with_imagenet.png", dpi=300, bbox_inches="tight")
#plt.show()

#%%
# New figure where we plot how many have the max product larger than x
import matplotlib.pyplot as plt
sns.set()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.figure(figsize=(10, 6))
plt.xlabel("x")
plt.ylabel("Fraction of CC entries with max larger than x")
xs = np.linspace(0, 1, 100)
ys = [np.sum(maxes > x) / len(maxes) for x in xs]
plt.plot(xs, ys)
plt.savefig("max_product_with_imagenet_fraction.png", dpi=300, bbox_inches="tight")
plt.show()








#%%
# Load the CC captions and get the urls and captions list
import pandas as pd
cc_captions_df = pd.read_csv(CC_CAPTIONS_DF, sep="\t", header=None)
cc_captions_df.columns = ["caption", "url"]
urls = cc_captions_df["url"].tolist()
captions = cc_captions_df["caption"].tolist()

#%%
print(captions[:100])

#%%
INTERESTING_CLASS = "chest"
# Find the id of that class
class_id = None
with open(IMAGENET_LABELS) as f:
    for i, line in enumerate(f):
        if INTERESTING_CLASS in line:
            class_id = i
            break
if class_id is None:
    raise ValueError(f"Class {INTERESTING_CLASS} not found in imagenet labels.")
print(f"Class {INTERESTING_CLASS} has id {class_id}")

# %%
# Find the CC entries that have that class as argmax
# and that have a max product larger than 0.5
THRESHOLD = 0.5 
interesting_entries = set()
for i, (entry, max_product) in enumerate(zip(argmax, maxes)):
    if entry == class_id and max_product > THRESHOLD:
        interesting_entries.add(i)

print(f"Found {len(interesting_entries)} entries with class {INTERESTING_CLASS} as argmax.")


# %%
# Get random 100 entries and their captions
import random
random.seed(42)
random_entries = random.sample(list(interesting_entries), min(40, len(interesting_entries)))

for entry in random_entries:
    print(f"{entry}: {captions[entry]}")

# %%

