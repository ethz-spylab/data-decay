# %%
import numpy as np
import pandas as pd
from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import numpy.lib.format
import struct
from torchvision import datasets, transforms
from contextlib import suppress
from open_clip import get_input_dtype

#%%
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

#%%
def get_relevant_captions_and_urls(similarity: np.ndarray, 
                          col_id : int, 
                          n_relevant = 10,
                          only_argmax = False,
                          sort_best = False,
                          seed: int = 42,
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
    
    cc = pd.read_csv(CAPTIONS_FILE, sep='\t', names=["caption","url"], usecols=range(0,2))
    captions = cc["caption"].tolist()
    urls = cc["url"].tolist()
    assert similarity.shape[0] == len(captions), "Similarity matrix and captions length do not match!"
    assert similarity.shape[1] - 1 >= abs(col_id), "col_id exceeds the # columns in similarity matrix!"
    similarity_relevant = similarity[:,col_id]
    if only_argmax == True:
        argmax = np.argmax(similarity, axis=1)
        similarity_relevant = similarity_relevant[argmax==col_id]
        captions = [captions[i] for i in np.where(argmax==col_id)[0]]
        urls = [urls[i] for i in np.where(argmax==col_id)[0]]
                              
    n_relevant_available = min(n_relevant, len(similarity_relevant))
    random.seed(seed)
                              
    if sort_best != True:
        random_entries = random.sample(range(len(similarity_relevant)), n_relevant_available)
        return [captions[entry] for entry in random_entries], [urls[entry] for entry in random_entries]
    else:
        idx = np.argpartition(similarity_relevant, -n_relevant_available)[-n_relevant_available:]
        idx_sorted = idx[np.argsort(similarity_relevant[idx])][::-1]
        return [captions[entry] for entry in idx_sorted], [urls[entry] for entry in idx_sorted]
    
# %%

def get_label_cluster_matching_captions_urls(label_assignment: np.ndarray,
                                             cluster_assignment: np.ndarray,
                                             relevant_labels: int,
                                             relevant_clusters: int,
                                             n_samples: int = 10,
                                             seed: int = 42,
                                             CAPTIONS_FILE = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"):
    """
    Returns the captions and urls of the given number of samples from the given label and cluster.
    """
    cc = pd.read_csv(CAPTIONS_FILE, sep='\t', names=["caption","url"], usecols=range(0,2))
    captions = cc["caption"].tolist()
    urls = cc["url"].tolist()

    label_cluster_cap = captions[(cluster_assignment == relevant_clusters) & (label_assignment == relevant_labels)]
    label_cluster_url = urls[(cluster_assignment == relevant_clusters) & (label_assignment == relevant_labels)]

    n_samples_available = min(n_samples, len(label_cluster_cap))

    random.seed(seed)
    random_entries = random.sample(range(len(label_cluster_cap)), n_samples_available)

    return label_cluster_cap[random_entries], label_cluster_url[random_entries]

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
    return get_relevant_captions_and_urls(similarity = comparison[:,None], 
                          col_id = 0, 
                          n_relevant = n_relevant,
                          sort_best = sort_best, 
                          CAPTIONS_FILE = CAPTIONS_FILE)

# %%
# TODO: create a function that generates number of missing bar plot and percentage of missing line plot for a given cluster

def plot_missing_num_perc(indices:np.ndarray, numbers: np.ndarray, percentages: np.ndarray, title="Plot", log_scale=True, x_label='class',
                y1_label='# of missing', y2_label='percentage of missing', labels=[]):
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
    if len(labels) == 0:
        ax1.bar([str(x) for x in indices],
                numbers[indices],
            color='g')
    elif len(labels) == len(indices):
        ax1.bar(labels,
                numbers[indices],
            color='g')
    else:
        raise ValueError("labels must be empty or same size as indices")
    ax1.set_ylabel(y1_label)
    ax1.set_xlabel(x_label)
    ax1.set_title(title)

    ax2 = plt.twinx()
    if len(labels) == 0:
        ax2.plot([str(x) for x in indices], 
            percentages[indices],
            color='b', marker='*')
    elif len(labels) == len(indices):
        ax2.plot(labels, 
            percentages[indices],
            color='b', marker='*')
    else:
        raise ValueError("labels must be empty or same size as indices")
    ax2.set_ylabel(y2_label)
    
    return fig

# %%
def get_cluster_make_up(cluster_number: int,
                        cluster_assignment: np.ndarray, 
                        label_assignment: np.ndarray, 
                        decayed_indices: list,
                        with_nan = False,
                        label_count = None):
    """
    Returns the make up of a cluster in terms of label assignments and the number of decayed elements in each assignment
        and the percentage of each label assignments in the cluster.
    
    Args:
        cluster_number: The number(index) of the cluster to be analyzed.
        cluster_assignment (size of dataset): The cluster assignment of each element in the dataset.
            cluster_assignment[i] : cluster number of element i
        label_assignment (size of CC dataset): The label assignment of each element in the dataset.
            label_assignment[i] : label of element i
        decayed_indices: The indices of the decayed elements in the dataset.
        with_nan: If True, the percentage of each label assignments in the cluster will be returned with nan values.
    Returns:
        cluster_label_element_counts: The number of elements for each label in the cluster.
        cluster_decayed_label_element_counts: The number of decayed elements for each label in the cluster.
        label_cluster_percentages: The percentage of each label assignment in the cluster.
            label_cluster_percentages[i] : # of elements with label i in the cluster / # of elements with label i in the dataset
            It has nan values in it!
        """
    
    if label_count is None:
        # Fails if no sample is assigned to the last label
        label_count = np.max(label_assignment) + 1

    cluster_indices = np.where(cluster_assignment == cluster_number)[0]
    cluster_label_assignment = label_assignment[cluster_indices]
    cluster_label_element_counts = np.zeros(label_count, dtype=int)
    for i in range(label_count):
        cluster_label_element_counts[i] = np.count_nonzero(cluster_label_assignment == i)

    decayed_array = np.zeros(label_assignment.shape[0], dtype=int)
    decayed_array[decayed_indices] = 1

    cluster_decayed_indices = np.where((cluster_assignment == cluster_number) & (decayed_array == 1))[0]
    cluster_decayed_label_assignment = label_assignment[cluster_decayed_indices]
    cluster_decayed_label_element_counts = np.zeros(label_count, dtype=int)

    for i in range(label_count):
        cluster_decayed_label_element_counts[i] = np.count_nonzero(cluster_decayed_label_assignment == i)

    label_element_counts = np.zeros(label_count, dtype=int)
    for i in range(label_count):
        label_element_counts[i] = np.count_nonzero(label_assignment == i)
        
    label_cluster_percentages = cluster_label_element_counts/(label_element_counts + 1e-6)
    
    if not with_nan:
        label_cluster_percentages[np.isnan(label_cluster_percentages)] = 0

    return cluster_label_element_counts, cluster_decayed_label_element_counts, label_cluster_percentages

# %%
def plot_cluster_make_up(cluster_number: int,
                         cluster_assignment: np.ndarray, 
                         label_assignment: np.ndarray, 
                         decayed_indices: list,
                         label_labels: list = [],
                         log_scale: bool = True,
                         x_label: str = 'Labels',
                         y1_label: str = '#',
                         y2_label: str = '% of a label in this cluster',
                         title: str = 'Make up of cluster',
                         order: str = "number",
                         n_plot: int = 10,
                         threshold_type: str = "num",
                         label_element_count_threshold : int = 100,
                         label_percentage_in_cluster_threshold : float = 0.5,
                         label_count : int = None):
    """
    Plots the make up of a cluster
    Args:
        cluster_number: The number(index) of the cluster to be analyzed.
        cluster_assignment (size of the dataset): The cluster assignment of each element in the dataset.
            cluster_assignment[i] : cluster number of element i
        label_assignment (size of CC dataset): The label assignment of each element in the dataset.
            label_assignment[i] : label of element i
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

    if label_count is None:
        # Fails if no sample is assigned to the last label
        label_count = np.max(label_assignment) + 1

    # get cluster make up
    cluster_label_element_counts, cluster_decayed_label_element_counts, \
        label_cluster_percentages = get_cluster_make_up(cluster_number, 
                                                        cluster_assignment, label_assignment, decayed_indices, label_count=label_count)
    if order == "number":
        indices = get_top_n_indices(cluster_label_element_counts,n_plot)
    elif order == "percentage":
        indices = get_top_n_indices(label_cluster_percentages,n_plot)
    elif order == "percentage_of_decayed":
        percentage_of_decayed = cluster_decayed_label_element_counts/cluster_label_element_counts
        percentage_of_decayed[np.isnan(percentage_of_decayed)] = 0
        percentage_of_decayed[cluster_label_element_counts < label_element_count_threshold] = 0
        # There might not be enough labels that satisfy the threshold in the cluster
        n_count = np.count_nonzero(percentage_of_decayed)
        indices = get_top_n_indices(percentage_of_decayed,min(n_plot,n_count))
    else:
        raise ValueError("order must be either 'number', 'percentage' or 'percentage_of_decayed'")
    # plot
    fig, ax1 = plt.subplots()
    if log_scale:
        ax1.set_yscale('log')
        y1_label = y1_label + ' (log scale)'
    plt.xticks(rotation = 45)
    if len(label_labels) > 0:
        labels = [label_labels[i]+" ("+str(i)+")" for i in indices]
    else:
        labels = [str(i) for i in indices]
    ax1.bar(labels,
            cluster_label_element_counts[indices],
            color='orange')
    c = ax1.bar(labels,
            cluster_decayed_label_element_counts[indices],
            color='g')
    bar_labels =[str("%2.f" % round(cluster_decayed_label_element_counts[i]/
                                    cluster_label_element_counts[i]*100, 2)) + "%" for i in indices]
    ax1.bar_label(c,bar_labels, color ='r')
    
    if title == 'Make up of cluster':
        title = title + ' ' + str(cluster_number) + ' ordered by ' + order

    #ax1.set_ylim(ymin=1)
    ax1.set_ylabel(y1_label)
    ax1.set_xlabel(x_label)
    ax1.set_title(title)
    l1 = plt.legend(['All', 'Decayed'], loc='upper right',framealpha=0.2)
    red_patch = mpatches.Patch(color='red', label='Decayed %')
    l2 = plt.legend(handles=[red_patch], loc='upper center',framealpha=0.2)
    ax1.add_artist(l1)
    ax1.add_artist(l2)

    ax2 = plt.twinx()
    ax2.plot(labels,
            label_cluster_percentages[indices]*100,
            color='b', marker='*')
    ax2.set_ylabel(y2_label)

    plt.show()

# %%
# TODO: for each cluster find what percentage of it is the top_k most common labels

def get_percentages_of_topk_labels_in_clusters(label_cluster_counts:np.ndarray, 
                          top_k: int = 3) -> np.ndarray:
    """
    Returns the summed percentage of top_k most common labels in each cluster.
        # of top_k most common labels in cluster i / # of elements in cluster i
    Args:
        label_cluster_counts: 2D array where rows are labels and columns are clusters.
            label_cluster_counts[i,j] : the number of elements in cluster j that have label i.
        top_k: number of most common labels to sum.
    Returns:
        label_cluster_percentages: 1D array same size as number of clusters.
            label_cluster_percentages[i] : the percentage of the top_k most common labels in cluster i."""
    
    number_of_clusters = label_cluster_counts.shape[1]
    label_cluster_percentages = np.zeros(number_of_clusters)
    cluster_counts = np.sum(label_cluster_counts, axis=0)
    for i in range(number_of_clusters):
        top_k_indices = get_top_n_indices(label_cluster_counts[:,i], top_k)
        label_cluster_percentages[i] = np.sum(label_cluster_counts[top_k_indices,i]) / cluster_counts[i]
    return label_cluster_percentages

# %%

def get_label_cluster_counts_and_decayed(cluster_assignment: np.ndarray,
                                            label_assignment: np.ndarray,
                                            decayed_indices: list = [],
                                            label_count: int = None):
    
    """For each label, count how many times it appears in each cluster and how many of them are decayed.
    Args:
        cluster_assignment: cluster assignment for each CC dataset element.
        label_assignment: label assignment for each CC dataset element.
        decayed_indices: list of indices of decayed CC dataset elements.
    Returns:
        label_cluster_counts: 2D array of shape (# labels, # clusters) where each row represents an label and each column represent a cluster.
        decayed_label_cluster_counts: 2D array of shape (# labels, # clusters) where each row represents an label and each column represent a cluster.
            Returned only if decayed_indices is not empty."""

    if label_count is None:
        # Fails if no sample is assigned to the last label
        label_count = np.max(label_assignment) + 1

    number_of_clusters = np.max(cluster_assignment) + 1
    label_cluster_counts = np.zeros((label_count,number_of_clusters), dtype=int) # rows are labels, columns are clusters
    for i in range(len(cluster_assignment)):
        label_cluster_counts[label_assignment[i], cluster_assignment[i]] += 1

    if len(decayed_indices) > 0:
        decayed_label_cluster_counts = np.zeros((label_count,number_of_clusters), dtype=int)
        for i in decayed_indices:
            decayed_label_cluster_counts[label_assignment[i], cluster_assignment[i]] += 1

        return label_cluster_counts, decayed_label_cluster_counts
    
    return label_cluster_counts, None

# %%
def plot_label_make_up(distribution_to_clusters : np.ndarray,
                          decayed_distribution_to_clusters : np.ndarray,
                          label_id : str = "",
                          label_name : str = "",
                          n_plot : int = 10):
    
    """Plot the distribution of labels in the clusters.
    Args:
        distribution_to_clusters: 1D array of shape (# clusters) where each element represents the number of labels in the cluster.
        decayed_distribution_to_clusters: 1D array of shape (# clusters) where each element represents the number of decayed labels in the cluster.
        title: label id as string.
        n_plot: number of clusters to plot. In addition to that the rest of the clusters are plotted as one bar.
    """
    
    top_indices = get_top_n_indices(distribution_to_clusters, n_plot)
    X_axis = [str(x) for x in top_indices]
    X_axis.append('Rest')

    fig, ax1 = plt.subplots()
    plt.xticks(rotation = 45)
    ax1.bar(X_axis, 
            np.append(distribution_to_clusters[top_indices], 
                              np.sum(distribution_to_clusters) - np.sum(distribution_to_clusters[top_indices])), 
            color='orange')
    c = ax1.bar(X_axis,
            np.append(decayed_distribution_to_clusters[top_indices], 
                              np.sum(decayed_distribution_to_clusters) - np.sum(decayed_distribution_to_clusters[top_indices])), 
            color='g')
    bar_labels =[str("%2.f" % round(decayed_distribution_to_clusters[i]/
                                    distribution_to_clusters[i]*100, 2)) + "%" for i in top_indices]
    bar_labels.append(str("%2.f" % round((np.sum(decayed_distribution_to_clusters) - np.sum(decayed_distribution_to_clusters[top_indices]))/
                                    (np.sum(distribution_to_clusters) - np.sum(distribution_to_clusters[top_indices]))*100, 2)) + "%")
    ax1.bar_label(c,bar_labels, color ='r')
    ax1.set_title('Distribution of label ' + label_id +"-" + 
              label_name + ' to the clusters')
    l1 = ax1.legend(['All', 'Decayed'], loc='upper right',framealpha=0.2)
    red_patch = mpatches.Patch(color='red', label='Decayed %')
    l2 = ax1.legend(handles=[red_patch], loc='upper center',framealpha=0.2)
    ax1.add_artist(l1)
    ax1.add_artist(l2)
    plt.show()

# %%

def find_matching_labels_and_clusters(cluster_assignment: np.ndarray, 
                                      label_assignment: np.ndarray,
                                      decayed_indices: list,
                                      labels_short = 'list',
                                      labels_long = 'list',
                                      label_element_count_threshold : int = 1000,
                                      label_percentage_in_cluster_threshold : float = 0.5,
                                      cluster_percentage_in_label_threshold : float = 0.4,
                                      decay_percentage_of_label_in_cluster_threshold : float = 0,
                                      plot : bool = True,
                                      summary : bool = True):
    
    """Find labels that are overrepresented in clusters and clusters that are overrepresented in labels.
    Args:
        cluster_assignment: cluster assignment for each CC dataset element.
        label_assignment: label assignment for each CC dataset element.
        decayed_indices: list of indices of decayed CC dataset elements.
        label_element_count_threshold: labels with less than this number of elements are ignored.
        label_percentage_in_cluster_threshold: labels with less than this percentage in a cluster are ignored.
        cluster_percentage_in_label_threshold: clusters with less than this percentage in an label are ignored.
        summary: if True, print a summary of the results.

    Returns:
        relevant_labels: list of labels that are overrepresented in clusters.
        relevant_clusters: list of clusters that are overrepresented in labels.
    """

    label_count = len(labels_short)
    
    label_cluster_counts, decayed_label_cluster_counts = get_label_cluster_counts_and_decayed(cluster_assignment, 
                                                                                                       label_assignment, 
                                                                                                       decayed_indices,
                                                                                                       label_count)
    

    print(f'label_cluster_counts shape: {label_cluster_counts.shape}')
    # for label_cluster_counts rows are labels, columns are clusters
    a = np.sum(label_cluster_counts, axis=1, keepdims=True).tolist()
    save_path = Path('/data/cc3m/script_tests/a.txt')
    with open(save_path, 'w') as fout:
        for i in a:
            fout.write(str(i) + '\n')

    distribution_of_labels_to_clusters = label_cluster_counts / (np.sum(label_cluster_counts, axis=1, keepdims=True) + 1e-6)
    # first row second column is the what percentage of label 0 is in cluster 1
    percentages_of_labels_in_clusters = label_cluster_counts / (np.sum(label_cluster_counts, axis=0, keepdims=True) + 1e-6)
    # first row second column is the what percentage of cluster 1 is label 0

    im_element_count = np.sum(label_cluster_counts, axis=1)
    decayed_im_element_count = np.sum(decayed_label_cluster_counts, axis=1)
    cl_element_count = np.sum(label_cluster_counts, axis=0)
    decayed_cl_element_count = np.sum(decayed_label_cluster_counts, axis=0)

    poss = np.column_stack(np.where(distribution_of_labels_to_clusters > label_percentage_in_cluster_threshold))
    rows = poss[:,0]
    cols = poss[:,1]

    relevant_labels = []
    relevant_clusters = []

    for row,col in zip(rows, cols):
        if (im_element_count[row] >= label_element_count_threshold) \
            and (percentages_of_labels_in_clusters[row, col] >= cluster_percentage_in_label_threshold \
            and (decayed_im_element_count[row]/im_element_count[row] >= decay_percentage_of_label_in_cluster_threshold)):
            relevant_labels.append(row)
            relevant_clusters.append(col)

    for i in range(len(relevant_clusters)):
        if plot:
            plot_cluster_make_up(relevant_clusters[i], cluster_assignment, 
                                 label_assignment, decayed_indices, 
                                 labels_short, order="number", label_count=label_count)
            plot_label_make_up(label_cluster_counts[relevant_labels[i],:], 
                                  decayed_label_cluster_counts[relevant_labels[i],:], 
                                  str(relevant_labels[i]),
                                  labels_short[relevant_labels[i]])
        if summary:
            print("label: (" + str(relevant_labels[i]) + ") " + 
                  labels_long[relevant_labels[i]] + 
                  ", cluster: " + str(relevant_clusters[i]))
            print(f'cluster in label: {label_cluster_counts[relevant_labels[i],relevant_clusters[i]]}, \
total cluster: {cl_element_count[relevant_clusters[i]]}, \
cluster percentage in label: {percentages_of_labels_in_clusters[relevant_labels[i],relevant_clusters[i]]:.3f}')
            print(f'label in cluster: {label_cluster_counts[relevant_labels[i],relevant_clusters[i]]}, \
total label: {im_element_count[relevant_labels[i]]}, \
label percentage in cluster: {distribution_of_labels_to_clusters[relevant_labels[i],relevant_clusters[i]]:.3f}')
            print(f'decay rate of label in cluster: \
{decayed_label_cluster_counts[relevant_labels[i],relevant_clusters[i]]/label_cluster_counts[relevant_labels[i],relevant_clusters[i]]:.3f}, \
decay rate of label in dataset: {decayed_im_element_count[relevant_labels[i]]/im_element_count[relevant_labels[i]]:.3f}, \
decay rate of cluster in dataset: {decayed_cl_element_count[relevant_clusters[i]]/cl_element_count[relevant_clusters[i]]:.3f}')
                  
    
    return relevant_labels, relevant_clusters

# %%
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

# %%
def dot_products_distances(emb_A, emb_B, device=torch.device('cuda:7')):
    """Compute the dot products between all pairs of vectors in emb_A and emb_B.
    Args:
        emb_A: np.array of shape (n_A, d)
        emb_B: np.array of shape (n_B, d)
    Returns:
        np.array of shape (n_A, n_B) with the dot products.
    """
    import torch
    emb_A = torch.from_numpy(emb_A).to(torch.float32).to(device)
    emb_B = torch.from_numpy(emb_B).to(torch.float32).to(device)
    dot_products = torch.mm(emb_A, emb_B.t()).cpu().numpy()
    distances = torch.cdist(emb_A, emb_B).cpu().numpy()
    return dot_products, distances

# %%

def create_dataloader(image_folder, batch_size=32, shuffle=False, num_workers=4, seed=42, transform=None):
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = datasets.ImageFolder(image_folder, transform=transform)
    torch.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

# %%

def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif precision == 'float32':
        return lambda: torch.cuda.amp.autocast(dtype=torch.float32)
    else:
        return suppress
    
# %%
def accuracy(output, target, topk=(1,)):
    """Computes top k accuracy
    Args:
        output (torch.Tensor): model output
        target (torch.Tensor): ground truth labels
        topk (tuple): top k values to calculate accuracy for
    Returns:
        list: top k accuracies"""
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

#%%
def get_accuracy_logits_targets(model, classifier, dataloader, args):
    """" Calculates accuracy, logits and targets for a given model, classifier and dataloader
    Args:
        model (torch.nn.Module): model to evaluate
        classifier (torch.nn.Module): classifier to use
        dataloader (torch.utils.data.DataLoader): dataloader to use
        args (Args): arguments
    Returns:
        float: top 1 accuracy
        float: top 5 accuracy
        list: logits
        list: targets
    """
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    tot_targets = []
    tot_logits = []
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        #for images, target in tqdm(dataloader, unit_scale=args.batch_size, leave=False):
        for images, target in dataloader:
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                """ output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0] """
                image_features = model.encode_image(images, normalize=True)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            tot_targets.extend(target)
            tot_logits.append(logits)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5, tot_logits, tot_targets

def get_precision_recall(preds, targets, length=None):
    """ Calculates precision and recall for each class
    Args:
        preds (np.array): predictions
        targets (np.array): targets
    Returns:
        np.array: precision
        np.array: recall"""
    
    if length is None:
        length = np.max(targets) + 1
    true_positives = np.zeros(length)
    true_and_false_positives = np.zeros(length)
    class_count = np.zeros(length)

    corrects = preds == targets

    for i in range(length):
        true_positives[i] = corrects[targets == i].sum()
        true_and_false_positives[i] = np.sum(preds == i)
        class_count[i]= (np.sum(targets == i))

    # It is possible a class is never predicted, in that case precision is nan
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = true_positives / true_and_false_positives
    recall = true_positives / class_count
    return precision, recall

def sort_list_by_occurences(list_to_sort):
    """"Sorts a list by the number of occurences of its elements in descending order.
    Returns a dictionary with the elements as keys and their number of occurences as values."""
    uniques = list(set(list_to_sort))
    occurrences = {}
    for item in uniques:
        count = list_to_sort.count(item)
        occurrences[item] = count
    sorted_occurrences = dict(sorted(occurrences.items(), key=lambda item: item[1], reverse=True))
    return sorted_occurrences

def get_diff_percent(recall_1, recall_2, nans_to_zero=True,
                     abs_diff=False):
    """Get the difference in percent between two recalls.
    If nans_to_zero is True, then nan values are set to 0.
    If abs_diff is True, then the recall_diff is multiplied by 50 and returned as recall_diff_abs.
    For each class, this value represents the difference in correctly classified elements between the two models."""
    recall_diff = recall_1 - recall_2
    recall_max = np.max(np.vstack((recall_1, recall_2)), axis=0)
    recall_diff_percent = recall_diff / recall_max
    if nans_to_zero:
        recall_diff_percent[np.isnan(recall_diff_percent)] = 0
    if abs_diff:
        recall_diff_abs = recall_diff*50
        return recall_diff_abs, recall_diff_percent
    return recall_diff_percent

# %%
def get_precision_recall_topk(logits, targets, topk = 5, length=None):

    """
    Computes the precision and recall for top k predictions.
    Args:
        logits: a 2d numpy array or torch tensor of logits
        targets: a 1d numpy array or torch tensor of targets
        topk: the top k predictions to consider
    Returns:
        topk_precision: a 1d numpy array of precision values for each class, when predicting topk sets of labels
        topk_recall: a 1d numpy array of recall values for each class, when predicting topk sets of labels
    """

    if length is None:
        length = np.max(targets) + 1
    
    if not torch.is_tensor(logits):
        logits = torch.from_numpy(logits)
    if not torch.is_tensor(targets):
        targets = torch.from_numpy(targets)

    topk_preds = logits.topk(topk, 1, True, True)[1] # (num_samples, topk)
    correct = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds)) # (num_samples, topk)

    true_positives = np.zeros(length) # (num_classes,)
    true_and_false_positives = np.zeros(length) # (num_classes,)
    class_count = np.zeros(length) # (num_classes,)

    topk_preds = topk_preds.cpu().numpy() # (num_samples, topk)
    correct = correct.cpu().numpy() # (num_samples, topk)
    targets = targets.cpu().numpy() # (num_samples,)

    for i in range(length):
        class_count[i] = (targets == i).sum() # number of elements in class i
        true_positives[i] = correct[targets == i].sum() # number of class i elements that are topk predicted as class i
        true_and_false_positives[i] = (topk_preds==i).sum() # number of elements that are topk predicted as class i

        with np.errstate(divide='ignore', invalid='ignore'):
            topk_precision = true_positives / true_and_false_positives
        topk_recall = true_positives / class_count
    
    return topk_precision, topk_recall