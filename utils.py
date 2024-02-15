# %%
import numpy as np
import json
import pandas as pd


# %%
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def load_captions(file_path):

    """Loads the captions from the file at the given path.
    Would depend on the file format. Here we used CC3M format."""

    captions_urls = pd.read_csv(file_path, sep="\t", header=None)
    captions_urls.columns = ["caption", "url"]
    captions = np.array(captions_urls["caption"])
    return captions
# %%
def get_top_n_indices(array : np.ndarray, n: int = 10, sort = True):
    """Returns the indices of the top n elements of an array.
    
    Args:
        array (np.array): The array to find the indices of the top n elements.
        n (int): The number of indices to return.
    Returns:
        np.array: The indices of the top n elements of the array."""
    if sort:
        #return np.argsort(array)[::-1][:n]
        temp = np.argpartition(array, -n)[-n:]
        return temp[np.argsort(array[temp])][::-1]
    else:
        return np.argpartition(array, -n)[-n:]