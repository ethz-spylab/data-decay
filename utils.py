import numpy as np
import pandas as pd
from pathlib import Path
import random

#%%
def get_relevant_captions(similarity: np.ndarray, 
                          col_id : int, 
                          n_relevant = 10,
                          only_argmax = False,
                          sort_best = False, 
                          CAPTIONS_FILE = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"):
    """
    Get the relevant captions for a given column of the dot products matrix.
    Args:
        similarity: (cc entries, classes) matrix. Each row corresponds to a CC entry and each column to a class.o
                    The entries are similarity values between the CC entry and the class.
                    Example: dot product matrix.
                    Example: 1 - distance matrix
        col_id: The column to get the relevant captions for.
        n_relevant: The number of relevant captions to return.
        only_argmax: If True, only consider the captions most similar to given col. If False, consider all captions.
        sort_best: If True, return the top n_relevant similarities. If False, choose randomly.
    
    Return:
        A list of n_relevant captions.
    """

    # TODO take a look at argmax_check.py file for inspiration
    
    captions = pd.read_csv(CAPTIONS_FILE, sep='\t', names=["caption","url"], usecols=range(0,2))["caption"].tolist()
    assert similarity.shape[0] == len(captions), "Similarity matrix and captions length do not match!"
    assert similarity.shape[1] - 1 >= abs(col_id), "col_id exceeds the # columns in similarity matrix!"
    similarity_relevant = similarity[:,col_id]
    if only_argmax == True:
        argmax = np.argmax(similarity, axis=1)
        similarity_relevant = similarity_relevant[argmax==col_id]
        captions = [captions[i] for i in np.where(argmax==col_id)[0]]
                              
    n_relevant_available = min(n_relevant, len(similarity_relevant))
                              
    if sort_best != True:
        random_entries = random.sample(range(len(similarity_relevant)), n_relevant_available)
        return [captions[entry] for entry in random_entries]
    else:
        idx = np.argpartition(similarity_relevant, -n_relevant_available)[-n_relevant_available:]
        idx_sorted = idx[np.argsort(similarity_relevant[idx])][::-1]
        return [captions[entry] for entry in idx_sorted]
    


#%%
def get_relevant_captions_from_embeddings(embeddings: np.ndarray, 
                                          query : np.ndarray,
                                          distance_function = "dot_product",
                                          n_relevant = 10,
                                          sort_best = False, 
                                          CAPTIONS_FILE = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"):
    """
    Get the relevant captions for a given query.
    Args:
        embeddings: (cc entries, embedding size) matrix. Each row is an embedding for a CC entry.
        query: (embedding size,) vector. The query to get the relevant captions for.
        distance_function: The distance function to use. Can be "dot_product" or "euclidean".
        n_relevant: The number of relevant captions to return.
        sort_best: If True, return the top n_relevant similarities. If False, choose randomly.
    
    Return:
        A list of n_relevant captions.
    """

    # TODO Compute the similarity
    
    if distance_function == "dot_product":
        comparison = embeddings @ query
    elif distance_function == "euclidean":
        diff = embeddings - query
        distance = np.linalg.norm(diff, axis=1)
        comparison = 1 - distance
    else:
        raise NotImplementedError("This distance method is not implemented yet.")

    # TODO call get_relevant_captions with col_id = 0
    return get_relevant_captions(similarity = comparison[:,None], 
                          col_id = 0, 
                          n_relevant = n_relevant,
                          sort_best = sort_best, 
                          CAPTIONS_FILE = CAPTIONS_FILE)

