# %%
import numpy as np
import json
import pandas as pd
from munch import Munch
from collections import Counter
import yaml

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

def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)
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

def caption_list_represent_with_counts(caption_list, max_diff_elems=5):
    """
    Given a list of captions, return a string representation with counts for repeated elements.
    The representation will only include up to `max_diff_elems` different elements for brevity.

    Args:
        caption_list (list of str): The list of captions to represent.
        max_diff_elems (int): The maximum number of different elements to include in the representation.

    Returns:
        str: A string representation of the list with counts for repeated elements.
    """
    # Count the occurrences of each caption
    caption_counts = Counter(caption_list)
    
    # Sort captions by count in descending order and take the top `max_diff_elems`
    most_common_captions = caption_counts.most_common(max_diff_elems)
    
    # Build the string representation
    representation = ", ".join([f'"{caption}" * {count}' for caption, count in most_common_captions])
    
    return representation