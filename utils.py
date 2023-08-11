import numpy as np
import pandas as pd
from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
IMAGENET_LABEL_COUNT = 1000

#%%
def get_top_n_indices(array : np.ndarray, n: int = 10):
    """Returns the indices of the top n elements of an array.
    
    Args:
        array (np.array): The array to find the indices of the top n elements.
        n (int): The number of indices to return.
    Returns:
        np.array: The indices of the top n elements of the array."""
    
    return np.argsort(array)[::-1][:n]

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
        emb_A = torch.from_numpy(query).to(torch.float32).to("cuda")
        emb_B = torch.from_numpy(embeddings).to(torch.float32).to("cuda")
        distances = torch.cdist(emb_A[None,:], emb_B).cpu().numpy()[0]
        comparison = 1 - distances
    else:
        raise NotImplementedError("This distance method is not implemented.")

    # TODO call get_relevant_captions with col_id = 0
    return get_relevant_captions(similarity = comparison[:,None], 
                          col_id = 0, 
                          n_relevant = n_relevant,
                          sort_best = sort_best, 
                          CAPTIONS_FILE = CAPTIONS_FILE)

# %%
# TODO: create a function that generates number of missing bar plot and percentage of missing line plot for a given cluster

def plot_missing_num_perc(indices:np.ndarray, numbers: np.ndarray, percentages: np.ndarray, title="Plot", log_scale=True, x_label='class',
                y1_label='# of missing', y2_label='percentage of missing'):
    """Plots the number of missing and percentage of missing for a given cluster.
    
    Args:
        indices : array of indices to plot
        numbers : number of missing
        percentages : percentage of missing
        title (str): title of the plot
        log_scale (bool): whether to use log scale for the number of missing
    Returns:
        fig: figure of the plot
        """
    fig, ax1 = plt.subplots()
    if log_scale:
        ax1.set_yscale('log')
    plt.xticks(rotation = 45)
    ax1.bar([str(x) for x in indices],
            numbers[indices],
           color='g')
    ax1.set_ylabel(y1_label)
    ax1.set_xlabel(x_label)
    ax1.set_title(title)

    ax2 = plt.twinx()
    ax2.plot([str(x) for x in indices], 
            percentages[indices],
            color='b', marker='*')
    ax2.set_ylabel(y2_label)
    
    return fig


# %%
def get_cluster_make_up(cluster_number: int,
                        cluster_assignment: np.ndarray, 
                        imagenet_assignment: np.ndarray, 
                        decayed_indices: list,
                        with_nan = False):
    """
    Returns the make up of a cluster in terms of imagenet assignments and the number of decayed elements in each assignment
        and the percentage of each imagenet label assignments in the cluster.
    
    Args:
        cluster_number: The number(index) of the cluster to be analyzed.
        cluster_assignment (size of CC dataset): The cluster assignment of each element in the dataset.
            cluster_assignment[i] : cluster number of element i
        imagenet_assignment (size of CC dataset): The imagenet assignment of each element in the dataset.
            imagenet_assignment[i] : imagenet label of element i
        decayed_indices: The indices of the decayed elements in the dataset.
        with_nan: If True, the percentage of each imagenet label assignments in the cluster will be returned with nan values.
    Returns:
        cluster_imagenet_element_counts: The number of elements for each imagenet label in the cluster.
        cluster_decayed_imagenet_element_counts: The number of decayed elements for each imagenet label in the cluster.
        imagenet_cluster_percentages: The percentage of each imagenet assignment in the cluster.
            imagenet_cluster_percentages[i] : # of elements with imagenet label i in the cluster / # of elements with imagenet label i in the dataset
            It has nan values in it!
        """
    cluster_indices = np.where(cluster_assignment == cluster_number)[0]
    cluster_imagenet_assignment = imagenet_assignment[cluster_indices]
    cluster_imagenet_element_counts = np.zeros(IMAGENET_LABEL_COUNT, dtype=int)
    for i in range(IMAGENET_LABEL_COUNT):
        cluster_imagenet_element_counts[i] = np.count_nonzero(cluster_imagenet_assignment == i)

    decayed_array = np.ones(imagenet_assignment.shape[0], dtype=int)
    decayed_array[decayed_indices] = 0

    cluster_decayed_indices = np.where((cluster_assignment == cluster_number) & (decayed_array == 0))[0]
    cluster_decayed_imagenet_assignment = imagenet_assignment[cluster_decayed_indices]
    cluster_decayed_imagenet_element_counts = np.zeros(IMAGENET_LABEL_COUNT, dtype=int)

    for i in range(IMAGENET_LABEL_COUNT):
        cluster_decayed_imagenet_element_counts[i] = np.count_nonzero(cluster_decayed_imagenet_assignment == i)

    imagenet_element_counts = np.zeros(IMAGENET_LABEL_COUNT, dtype=int)
    for i in range(IMAGENET_LABEL_COUNT):
        imagenet_element_counts[i] = np.count_nonzero(imagenet_assignment == i)
        
    imagenet_cluster_percentages = cluster_imagenet_element_counts/imagenet_element_counts
    
    if not with_nan:
        imagenet_cluster_percentages[np.isnan(imagenet_cluster_percentages)] = 0

    return cluster_imagenet_element_counts, cluster_decayed_imagenet_element_counts, imagenet_cluster_percentages


# %%
def plot_cluster_make_up(cluster_number: int,
                         cluster_assignment: np.ndarray, 
                         imagenet_assignment: np.ndarray, 
                         decayed_indices: list,
                         imagenet_labels: list = [],
                         log_scale: bool = True,
                         x_label: str = 'Imagenet Labels',
                         y1_label: str = '#',
                         y2_label: str = 'percentage of a label in this cluster',
                         title: str = 'Make up of cluster',
                         order: str = "number",
                         n_plot: int = 10):
    """
    Plots the make up of a cluster
    Args:
        cluster_number: The number(index) of the cluster to be analyzed.
        cluster_assignment (size of CC dataset): The cluster assignment of each element in the dataset.
            cluster_assignment[i] : cluster number of element i
        imagenet_assignment (size of CC dataset): The imagenet assignment of each element in the dataset.
            imagenet_assignment[i] : imagenet label of element i
        decayed_indices: The indices of the decayed elements in the dataset.
        log_scale: Whether to use log scale for the y axis.
        x_label: The label of the x axis.
        y1_label: The label of the y1 axis.
        y2_label: The label of the y2 axis.
        title: The title of the plot.
        order: The order of the bars in the plot. Either "number" or "percentage".
        n_plot: The number of bars to plot.
    Returns:
        None
    """
    # get cluster make up
    cluster_imagenet_element_counts, cluster_decayed_imagenet_element_counts, imagenet_cluster_percentages = get_cluster_make_up(cluster_number, cluster_assignment, imagenet_assignment, decayed_indices)
    if order == "number":
        indices = get_top_n_indices(cluster_imagenet_element_counts,n_plot)
    elif order == "percentage":
        indices = get_top_n_indices(imagenet_cluster_percentages,n_plot)
    else:
        raise ValueError("order must be either 'number' or 'percentage'")
    # plot
    fig, ax1 = plt.subplots()
    if log_scale:
        ax1.set_yscale('log')
        y1_label = y1_label + ' (log scale)'
    plt.xticks(rotation = 45)
    if len(imagenet_labels) > 0:
        labels = [imagenet_labels[i]+" ("+str(i)+")" for i in indices]
    else:
        labels = [str(i) for i in indices]
    ax1.bar(labels,
            cluster_imagenet_element_counts[indices],
            color='g')
    ax1.bar(labels,
            cluster_decayed_imagenet_element_counts[indices],
            color='orange')
    
    if title == 'Make up of cluster':
        title = title + ' ' + str(cluster_number) + ' ordered by ' + order

    ax1.set_ylabel(y1_label)
    ax1.set_xlabel(x_label)
    ax1.set_title(title)
    ax1.legend(['All', 'Decayed'])

    ax2 = plt.twinx()
    ax2.plot(labels,
            imagenet_cluster_percentages[indices],
            color='b', marker='*')
    ax2.set_ylabel(y2_label)

    plt.show()


# %%
# TODO: for each cluster find what percentage of it is the top_k most common imagenet labels

def get_percentages_of_topk_labels_in_clusters(imagenet_cluster_counts:np.ndarray, 
                          top_k: int = 3) -> np.ndarray:
    """
    Returns the summed percentage of top_k most common imagenet labels in each cluster.
        # of top_k most common imagenet labels in cluster i / # of elements in cluster i
    Args:
        imagenet_cluster_counts: 2D array where rows are imagenet labels and columns are clusters.
            imagenet_cluster_counts[i,j] : the number of elements in cluster j that have imagenet label i.
        top_k: number of most common imagenet labels to sum.
    Returns:
        imagenet_cluster_percentages: 1D array same size as number of clusters.
            imagenet_cluster_percentages[i] : the percentage of the top_k most common imagenet labels in cluster i."""
    
    number_of_clusters = imagenet_cluster_counts.shape[1]
    imagenet_cluster_percentages = np.zeros(number_of_clusters)
    cluster_counts = np.sum(imagenet_cluster_counts, axis=0)
    for i in range(number_of_clusters):
        top_k_indices = get_top_n_indices(imagenet_cluster_counts[:,i], top_k)
        imagenet_cluster_percentages[i] = np.sum(imagenet_cluster_counts[top_k_indices,i]) / cluster_counts[i]
    return imagenet_cluster_percentages


# %%
# TODO: for each imagenet label find what percentage of it is in each cluster

def get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts:np.ndarray) -> np.ndarray:
    """
    Returns the percentage of each imagenet label in each cluster.
        # of imagenet label i in cluster j / # of imagenet label i
    Args:
        imagenet_cluster_counts: 2D array of # elements where rows are imagenet labels and columns are clusters.
            imagenet_cluster_counts[i,j] : the number of elements in cluster j that have imagenet label i.
    Returns:
        imagenet_cluster_percentages: 2D array of percentages where rows are imagenet labels and columns are clusters.
            imagenet_cluster_percentages[i,j] : the percentage of imagenet label i in cluster j to the total number of imagenet label i."""
    
    return imagenet_cluster_counts / np.sum(imagenet_cluster_counts, axis=1, keepdims=True)


# %%

def get_imagenet_cluster_counts_and_decayed(cluster_assignment: np.ndarray,
                                            imagenet_assignment: np.ndarray,
                                            decayed_indices: list = []):
    
    """For each imagenet labels, count how many times it appears in each cluster and how many of them are decayed.
    Args:
        cluster_assignment: cluster assignment for each CC dataset element.
        imagenet_assignment: imagenet label assignment for each CC dataset element.
        decayed_indices: list of indices of decayed CC dataset elements.
    Returns:
        imagenet_cluster_counts: 2D array of shape (# imagenet labels, # clusters) where each row represents an imagenet label and each column represent a cluster.
        decayed_imagenet_cluster_counts: 2D array of shape (# imagenet labels, # clusters) where each row represents an imagenet label and each column represent a cluster.
            Returned only if decayed_indices is not empty."""

    number_of_clusters = np.max(cluster_assignment) + 1
    imagenet_cluster_counts = np.zeros((IMAGENET_LABEL_COUNT,number_of_clusters), dtype=int)
    for i in range(len(cluster_assignment)):
        imagenet_cluster_counts[imagenet_assignment[i], cluster_assignment[i]] += 1

    if len(decayed_indices) > 0:
        decayed_imagenet_cluster_counts = np.zeros((IMAGENET_LABEL_COUNT,number_of_clusters), dtype=int)
        for i in decayed_indices:
            decayed_imagenet_cluster_counts[imagenet_assignment[i], cluster_assignment[i]] += 1

        return imagenet_cluster_counts, decayed_imagenet_cluster_counts
    
    return imagenet_cluster_counts


def plot_imagenet_make_up(distribution_to_clusters : np.ndarray,
                          decayed_distribution_to_clusters : np.ndarray,
                          imagenet_label_id : str = "",
                          imagenet_label_name : str = "",
                          n_plot : int = 10):
    
    """Plot the distribution of imagenet labels in the clusters.
    Args:
        distribution_to_clusters: 1D array of shape (# clusters) where each element represents the number of imagenet labels in the cluster.
        decayed_distribution_to_clusters: 1D array of shape (# clusters) where each element represents the number of decayed imagenet labels in the cluster.
        title: imagenet label id as string.
        n_plot: number of clusters to plot. In addition to that the rest of the clusters are plotted as one bar.
    """
    
    top_indices = get_top_n_indices(distribution_to_clusters, n_plot)
    X_axis = [str(x) for x in top_indices]
    X_axis.append('Rest')

    plt.xticks(rotation = 45)
    plt.bar(X_axis, 
            np.append(distribution_to_clusters[top_indices], 
                              np.sum(distribution_to_clusters) - np.sum(distribution_to_clusters[top_indices])), 
            color='g')
    plt.bar(X_axis,
            np.append(decayed_distribution_to_clusters[top_indices], 
                              np.sum(decayed_distribution_to_clusters) - np.sum(decayed_distribution_to_clusters[top_indices])), 
            color='orange')
    plt.title('Distribution of imagenet label ' + imagenet_label_id +"-" + 
              imagenet_label_name+ ' to the clusters')
    plt.legend(['All', 'Decayed'])
    plt.show()    


# %%

def find_matching_labels_and_clusters(cluster_assignment: np.ndarray, 
                                      imagenet_assignment: np.ndarray,
                                      decayed_indices: list,
                                      imagenet_labels_short = 'list',
                                      imagenet_labels_long = 'list',
                                      imagenet_element_count_threshold : int = 1000,
                                      imagenet_percentage_in_cluster_threshold : float = 0.5,
                                      cluster_percentage_in_imagenet_threshold : float = 0.4,
                                      plot : bool = True,
                                      summary : bool = True):
    
    """Find imagenet labels that are overrepresented in clusters and clusters that are overrepresented in imagenet labels.
    Args:
        cluster_assignment: cluster assignment for each CC dataset element.
        imagenet_assignment: imagenet label assignment for each CC dataset element.
        decayed_indices: list of indices of decayed CC dataset elements.
        imagenet_element_count_threshold: imagenet labels with less than this number of elements are ignored.
        imagenet_percentage_in_cluster_threshold: imagenet labels with less than this percentage in a cluster are ignored.
        cluster_percentage_in_imagenet_threshold: clusters with less than this percentage in an imagenet label are ignored.
        summary: if True, print a summary of the results.

    Returns:
        relevant_labels: list of imagenet labels that are overrepresented in clusters.
        relevant_clusters: list of clusters that are overrepresented in imagenet labels.
    """
    
    imagenet_cluster_counts, decayed_imagenet_cluster_counts = get_imagenet_cluster_counts_and_decayed(cluster_assignment, imagenet_assignment, decayed_indices)

    distribution_of_imagenet_labels_to_clusters = imagenet_cluster_counts / np.sum(imagenet_cluster_counts, axis=1, keepdims=True)
    percentages_of_imagenet_labels_in_clusters = imagenet_cluster_counts / np.sum(imagenet_cluster_counts, axis=0, keepdims=True)

    im_element_count = np.sum(imagenet_cluster_counts, axis=1)
    decayed_im_element_count = np.sum(decayed_imagenet_cluster_counts, axis=1)
    cl_element_count = np.sum(imagenet_cluster_counts, axis=0)
    decayed_cl_element_count = np.sum(decayed_imagenet_cluster_counts, axis=0)

    poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > imagenet_percentage_in_cluster_threshold))
    rows = poss[:,0]
    cols = poss[:,1]

    relevant_labels = []
    relevant_clusters = []

    for row,col in zip(rows, cols):
        if (im_element_count[row] >= imagenet_element_count_threshold) & (percentages_of_imagenet_labels_in_clusters[row, col] >= cluster_percentage_in_imagenet_threshold):
            relevant_labels.append(row)
            relevant_clusters.append(col)

    for i in range(len(relevant_clusters)):
        if plot:
            plot_cluster_make_up(relevant_clusters[i], cluster_assignment, 
                                 imagenet_assignment, decayed_indices, 
                                 imagenet_labels_short, order="number")
            plot_imagenet_make_up(imagenet_cluster_counts[relevant_labels[i],:], 
                                  decayed_imagenet_cluster_counts[relevant_labels[i],:], 
                                  str(relevant_labels[i]),
                                  imagenet_labels_short[relevant_labels[i]])
        if summary:
            print("label: (" + str(relevant_labels[i]) + ") " + 
                  imagenet_labels_long[relevant_labels[i]] + 
                  ", cluster: " + str(relevant_clusters[i]))
            print("cluster in label: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                  ", total cluster: ", cl_element_count[relevant_clusters[i]],
                  ", cluster percentage in label: ", percentages_of_imagenet_labels_in_clusters[relevant_labels[i],relevant_clusters[i]])            
            print("label in cluster: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                  ", total label: ", im_element_count[relevant_labels[i]],
                  ", label percentage in cluster: ", distribution_of_imagenet_labels_to_clusters[relevant_labels[i],relevant_clusters[i]])
            #print("decay rate of label in cluster: ", decayed_imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]]/imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
            #      ", decay rate of label in dataset: ", decayed_im_element_count[relevant_labels[i]]/im_element_count[relevant_labels[i]],
            #      ", decay rate of cluster in dataset: ", decayed_cl_element_count[relevant_clusters[i]]/cl_element_count[relevant_clusters[i]])
            print(f'decay rate of label in cluster: {decayed_imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]]/imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]]:.3f}, \
decay rate of label in dataset: {decayed_im_element_count[relevant_labels[i]]/im_element_count[relevant_labels[i]]:.3f}, \
decay rate of cluster in dataset: {decayed_cl_element_count[relevant_clusters[i]]/cl_element_count[relevant_clusters[i]]:.3f}')
                  
    
    return relevant_labels, relevant_clusters
