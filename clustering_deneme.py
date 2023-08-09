#%%
from pathlib import Path
DATA_FOLDER = Path("/data/cc3m")
EMBEDDINGS_FOLDER = DATA_FOLDER / "cc3m_2023/embeddings"
CC_EMBEDDINGS_FOLDER = EMBEDDINGS_FOLDER / "text_embeddings.npy"
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"
CC_VS_IMAGENET = EMBEDDINGS_FOLDER / "CC_vs_imagenet_L14.npy"
DECAYED_INDICES = DATA_FOLDER / "decayed_indices.txt"
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import (plot_missing_num_perc, get_relevant_captions, get_cluster_make_up, plot_cluster_make_up,  
    get_top_n_indices, get_percentages_of_topk_labels_in_clusters, 
    get_distribution_of_imagenet_labels_to_clusters, get_imagenet_cluster_counts_and_decayed,
    plot_imagenet_make_up, find_matching_labels_and_clusters)
import pickle
import torch
CLUSTER_CENTERS = EMBEDDINGS_FOLDER / "cluster_centers.npy"
DOT_PRODUCTS = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers.npy"
DISTANCES = EMBEDDINGS_FOLDER / "CC_vs_cluster_centers_distances.npy"

IMAGENET_LABEL_COUNT = 1000

# %%
cluster_centers = np.load(CLUSTER_CENTERS)
dot_products = np.load(DOT_PRODUCTS)
distances = np.load(DISTANCES)

# %%
# TODO: check if dot products make sense
get_relevant_captions(dot_products, 63, only_argmax=True, sort_best=False)

# %%
cluster_assignment = np.argmax(dot_products, axis=1)
print(cluster_assignment.shape)

# %%
number_of_clusters = dot_products.shape[1]
# %%
# TODO: find how many elements are there in each cluster
cluster_element_counts = np.zeros(number_of_clusters)
for i in range(number_of_clusters):
    cluster_element_counts[i] = np.count_nonzero(cluster_assignment == i)

# %%
#TODO: plot the distribution of the clusters

plt.hist(cluster_element_counts, bins=number_of_clusters)
plt.title('Distribution of the clusters')
plt.show()

# %%
# TODO: plot the 10 clusters with the highest number of elements in log scale
highest_number_clusters = get_top_n_indices(cluster_element_counts,10)
plt.bar([str(x) for x in highest_number_clusters], cluster_element_counts[highest_number_clusters])
plt.yscale('log')
plt.title('10 clusters with highest number of elements')
plt.show()


# %% 
# TODO: read decayed indices to a list
decayed_indices = []
with open(DECAYED_INDICES, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))
print("Number of decayed indices: ", len(decayed_indices))

# %%
cc_captions = pd.read_csv(CC_CAPTIONS_DF, sep="\t", header=None)
cc_captions.columns = ["caption", "url"]
captions = cc_captions["caption"].tolist()
url = cc_captions["url"].tolist()

# %%
# TODO: list first 10 urls of decayed indices
print("First 10 urls of decayed indices:")
for i in range(10):
    print(url[decayed_indices[i]])

# %%
# TODO: find percentage of decayed indices in each cluster
decayed_in_clusters = np.zeros(number_of_clusters, dtype=int)
for i in range(number_of_clusters):
    decayed_in_clusters[i] = np.count_nonzero(cluster_assignment[decayed_indices] == i)
percentage_of_decayed_indices_in_clusters = decayed_in_clusters / cluster_element_counts

# %%
# TODO: bar plot the percentage of decayed indices for each cluster
plt.bar(np.arange(number_of_clusters), percentage_of_decayed_indices_in_clusters)
plt.title('Percentage of decayed indices in each cluster')
plt.show()

# %%
# TODO: find the 10 clusters with the highest percentage of decayed indices and print them and their percentages
highest_percentage_clusters = get_top_n_indices(percentage_of_decayed_indices_in_clusters,10)
print(highest_percentage_clusters)
print(percentage_of_decayed_indices_in_clusters[highest_percentage_clusters])

# %%
# TODO: Plot the highest_percentage_clusters and their percentages, and the number of decayed indices in them in the same plot
plt.bar([str(x) for x in highest_percentage_clusters], percentage_of_decayed_indices_in_clusters[highest_percentage_clusters])
plt.title('Clusters with highest percentage of missing')
plt.show()

# %%

fig = plot_missing_num_perc(highest_percentage_clusters,
                            decayed_in_clusters, 
                            percentage_of_decayed_indices_in_clusters,
                            title='Clusters with highest percentage of missing')
plt.show()

# %%
# TODO: find the 10 clusters with the highest number of decayed indices
highest_number_decayed = get_top_n_indices(decayed_in_clusters)

fig = plot_missing_num_perc(highest_number_decayed, 
                   decayed_in_clusters, 
                   percentage_of_decayed_indices_in_clusters, 
                   'Clusters with highest number of missing')
plt.show()


# %%
cc_vs_imagenet = np.load(CC_VS_IMAGENET)
# %%
cc_vs_imagenet.shape
# %%
imagenet_assignment = np.argmax(cc_vs_imagenet, axis=1)
imagenet_assignment.shape

# %%

decayed_array = np.ones(imagenet_assignment.shape[0], dtype=int)
decayed_array[decayed_indices] = 0

print(type(decayed_array))
print(type(decayed_indices))


# %%
# TODO: find how many elements are there in each imagenet assignment
imagenet_element_counts = np.zeros(IMAGENET_LABEL_COUNT)
for i in range(IMAGENET_LABEL_COUNT):
    imagenet_element_counts[i] = np.count_nonzero(imagenet_assignment == i)


# TODO: find the number of imagenet_element_counts with 0 elements
print("Number of imagenet_element_counts with 0 elements: ", np.count_nonzero(imagenet_element_counts == 0))


#%%

plot_cluster_make_up(30, cluster_assignment, imagenet_assignment, decayed_indices, order="number")
plot_cluster_make_up(30, cluster_assignment, imagenet_assignment, decayed_indices, order="percentage")

# %%

plot_cluster_make_up(29, cluster_assignment, imagenet_assignment, decayed_indices, order="number")
plot_cluster_make_up(29, cluster_assignment, imagenet_assignment, decayed_indices, order="percentage")

# %%
# TODO: find cluster to caption assignment and print the caption for each cluster

cluster_to_caption_assignment = np.argmax(dot_products, axis=0)
for i in range(number_of_clusters):
    print("Cluster: ", i, " Match: ", 
          dot_products[cluster_to_caption_assignment[i],i],
          " # elements", int(cluster_element_counts[i]),
          " Caption: ", captions[cluster_to_caption_assignment[i]])
    
# %%

relevant_labels, relevant_clusters = find_matching_labels_and_clusters(cluster_assignment, imagenet_assignment, decayed_indices)












# AFTER THIS POINT IS NOT USED






















# %%

# %%
# TODO: find the number of elements in each imagenet label and cluster

imagenet_cluster_counts = np.zeros((IMAGENET_LABEL_COUNT,number_of_clusters), dtype=int)
decayed_imagenet_cluster_counts = np.zeros((IMAGENET_LABEL_COUNT,number_of_clusters), dtype=int)
for i in range(len(cluster_assignment)):
    imagenet_cluster_counts[imagenet_assignment[i], cluster_assignment[i]] += 1

for i in decayed_indices:
    decayed_imagenet_cluster_counts[imagenet_assignment[i], cluster_assignment[i]] += 1

# %%

decayed_in_imagenet = np.sum(decayed_imagenet_cluster_counts, axis=1)
print(decayed_in_imagenet.shape)

# %%
k = np.sum(imagenet_cluster_counts, axis=0)
l = np.sum(imagenet_cluster_counts, axis=1)
print(np.sum(k == cluster_element_counts))
print(np.sum(l == imagenet_element_counts))

# %%
percentages_of_top_labels_in_clusters = get_percentages_of_topk_labels_in_clusters(imagenet_cluster_counts, 1)
percentages_of_top_labels_in_clusters.shape

# %%
percentages_of_top_clusters_for_labels = get_percentages_of_topk_labels_in_clusters(imagenet_cluster_counts.T, 1)
percentages_of_top_clusters_for_labels.shape

# %%
print("max: ", np.nanmax(percentages_of_top_clusters_for_labels), " argmax: " , np.nanargmax(percentages_of_top_clusters_for_labels))
plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[np.nanargmax(percentages_of_top_clusters_for_labels),:])
plt.title('Distribution of imagenet label ' + str(np.nanargmax(percentages_of_top_clusters_for_labels)) + ' in the clusters')
plt.show()

#%%
large_ones = percentages_of_top_clusters_for_labels[imagenet_element_counts > 10000]
large_indices = get_top_n_indices(large_ones, 3)
rel = [np.where(percentages_of_top_clusters_for_labels == large_ones[x])[0][0] for x in large_indices]

for r in rel:
    plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[r,:])
    plt.title('Distribution of imagenet label ' + str(r) + ' in the clusters')
    plt.show()

# %%
#ust limit de konulabilir mesela suan surekli cluster 74 cikio
#decay olanlari da incele
#imagenet label distribution i function hala getirebiliriz

# %%
inds = get_top_n_indices(percentages_of_top_labels_in_clusters, 3)

for ind in inds:
    plot_cluster_make_up(ind, cluster_assignment, imagenet_assignment, decayed_indices, order="number")

# %%

for ind in inds:
    print("ind: ", ind, "cluster element #: ", cluster_element_counts[ind], " top %: ", percentages_of_top_labels_in_clusters[ind])


# %%

for ind in inds:
    print(np.argmax(imagenet_cluster_counts[:,ind]))
    print(imagenet_cluster_counts[np.argmax(imagenet_cluster_counts[:,ind]),ind])

# %%

distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)
#maxa = np.nanmax(a, axis=1)

#%%
print(distribution_of_imagenet_labels_to_clusters.shape)
print(percentages_of_imagenet_labels_in_clusters.shape)
np.sum(percentages_of_imagenet_labels_in_clusters, axis=1)

#%%

poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > 0.5))
rows = poss[:,0]
cols = poss[:,1]

#%%

imagenet_element_count_threshold = 1000
imagenet_percentage_in_cluster_threshold = 0.5
cluster_percentage_in_imagenet_threshold = 0.4

distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)

im_element_count = np.sum(imagenet_cluster_counts, axis=1)
cl_element_count = np.sum(imagenet_cluster_counts, axis=0)

poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > imagenet_percentage_in_cluster_threshold))
rows = poss[:,0]
cols = poss[:,1]

relevant_labels = []
relevant_clusters = []

for row,col in zip(rows, cols):
    if (im_element_count[row] >= imagenet_element_count_threshold) & (percentages_of_imagenet_labels_in_clusters[col, row] >= cluster_percentage_in_imagenet_threshold):
        relevant_labels.append(row)
        relevant_clusters.append(col)

summary = True

for i in range(len(relevant_clusters)):
    plot_cluster_make_up(relevant_clusters[i], cluster_assignment, imagenet_assignment, decayed_indices, order="number")
    plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[relevant_labels[i],:])
    plt.title('Distribution of imagenet label ' + str(relevant_labels[i]) + ' in the clusters')
    plt.show()
    if summary:
        print("label: ", relevant_labels[i], " cluster: ", relevant_clusters[i])
        print("label in cluster: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
              " total label: ", im_element_count[relevant_labels[i]],
              " label percentage in cluster: ", distribution_of_imagenet_labels_to_clusters[relevant_labels[i],relevant_clusters[i]])
        print("cluster in label: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                " total cluster: ", cl_element_count[relevant_clusters[i]],
                " cluster percentage in label: ", percentages_of_imagenet_labels_in_clusters[relevant_clusters[i],relevant_labels[i]])
    



    

# %%

print(percentages_of_imagenet_labels_in_clusters[79,879])
print(distribution_of_imagenet_labels_to_clusters[879,79])

# %%

percentages_of_imagenet_labels_in_clusters[79,879]
distribution_of_imagenet_labels_to_clusters[879,79]


plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[879,:])
plt.title('Distribution of imagenet label ' + str(879) + ' in the clusters')
plt.show()

plot_cluster_make_up(79, cluster_assignment, imagenet_assignment, decayed_indices, order="number")

# %%

np.sum(np.max(distribution_of_imagenet_labels_to_clusters, axis=1) == percentages_of_top_clusters_for_labels)
# nan is not equal to nan

#%%

np.where(imagenet_element_counts == 0)


# %%
#TODO: plot the distribution of imagenet label i in the clusters
for ind in inds:
    plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[np.argmax(imagenet_cluster_counts[:,ind]),:])
    plt.title('Distribution of imagenet label ' + str(ind) + ' in the clusters')
    plt.show()


# %%
rel
# %%
cluster_number = 10
cluster_indices = np.where(cluster_assignment == cluster_number)[0]
# %%
cluster_indices
# %%

imagenet_element_count_threshold = 1000
imagenet_percentage_in_cluster_threshold = 0.5
cluster_percentage_in_imagenet_threshold = 0.4

distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)

im_element_count = np.sum(imagenet_cluster_counts, axis=1)
cl_element_count = np.sum(imagenet_cluster_counts, axis=0)

poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > imagenet_percentage_in_cluster_threshold))
rows = poss[:,0]
cols = poss[:,1]

relevant_labels = []
relevant_clusters = []


row_counter = 0

for row in rows:
    row_counter += 1
    col_counter = 0
    for col in cols:
        col_counter += 1
        if (im_element_count[row] >= imagenet_element_count_threshold) & (percentages_of_imagenet_labels_in_clusters[col, row] >= cluster_percentage_in_imagenet_threshold):
            print("Adding label: ", row, " to cluster: ", col)
            print("Row counter: ", row_counter, " col counter: ", col_counter)
            relevant_labels.append(row)
            relevant_clusters.append(col)
        else:
            continue
# %%
print(relevant_labels)
print(relevant_clusters)

# %%
rows[76]
# %%
a = [1,2,3,4,5]
b = [1,2,3,4,5]

for x,y in zip(a,b):
    print(x,y)
# %%

def get_distribution_of_imagenet_labels_to_clusters2(imagenet_cluster_counts):
    return imagenet_cluster_counts / np.sum(imagenet_cluster_counts, axis=1, keepdims=True)
# %%
distribution_of_imagenet_labels_to_clusters2 = get_distribution_of_imagenet_labels_to_clusters2(imagenet_cluster_counts)
# %%
distribution_of_imagenet_labels_to_clusters2[879,79]
# %%
np.sum(imagenet_cluster_counts, axis=1, keepdims=True).shape
# %%
np.sum(imagenet_cluster_counts, axis=1).shape


# %%

imagenet_element_count_threshold = 1000
imagenet_percentage_in_cluster_threshold = 0.5
cluster_percentage_in_imagenet_threshold = 0.4

distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)

im_element_count = np.sum(imagenet_cluster_counts, axis=1)
cl_element_count = np.sum(imagenet_cluster_counts, axis=0)

poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > imagenet_percentage_in_cluster_threshold))
rows = poss[:,0]
cols = poss[:,1]

relevant_labels = []
relevant_clusters = []

for row,col in zip(rows, cols):
    if (im_element_count[row] >= imagenet_element_count_threshold) & (percentages_of_imagenet_labels_in_clusters[col, row] >= cluster_percentage_in_imagenet_threshold):
        relevant_labels.append(row)
        relevant_clusters.append(col)

summary = True

for i in range(len(relevant_clusters)):
    plot_cluster_make_up(relevant_clusters[i], cluster_assignment, imagenet_assignment, decayed_indices, order="number")
    plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[relevant_labels[i],:])
    plt.title('Distribution of imagenet label ' + str(relevant_labels[i]) + ' to the clusters')
    plt.show()
    if summary:
        print("label: ", relevant_labels[i], " cluster: ", relevant_clusters[i])
        print("label in cluster: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
              " total label: ", im_element_count[relevant_labels[i]],
              " label percentage in cluster: ", distribution_of_imagenet_labels_to_clusters[relevant_labels[i],relevant_clusters[i]])
        print("cluster in label: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                " total cluster: ", cl_element_count[relevant_clusters[i]],
                " cluster percentage in label: ", percentages_of_imagenet_labels_in_clusters[relevant_clusters[i],relevant_labels[i]])
    

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


#%%
def plot_imagenet_make_up(distribution_to_clusters : np.ndarray,
                          decayed_distribution_to_clusters : np.ndarray,
                          title : str = "",
                          n_plot : int = 10,
                          number_of_clusters : int = 100):
    
    top_indices = get_top_n_indices(distribution_to_clusters, n_plot)
    plt.bar([str(x) for x in top_indices], distribution_to_clusters[top_indices], color='g')
    plt.bar([str(x) for x in top_indices], decayed_distribution_to_clusters[top_indices], color='orange')
    plt.title('Distribution of imagenet label ' + title + ' in the clusters')
    plt.legend(['All', 'Decayed'])
    plt.show()    


#%%
def plot_imagenet_make_up(distribution_to_clusters : np.ndarray,
                          decayed_distribution_to_clusters : np.ndarray,
                          imagenet_label_id : str = "",
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

    plt.bar(X_axis, 
            np.append(distribution_to_clusters[top_indices], 
                              np.sum(distribution_to_clusters) - np.sum(distribution_to_clusters[top_indices])), 
            color='g')
    plt.bar(X_axis,
            np.append(decayed_distribution_to_clusters[top_indices], 
                              np.sum(decayed_distribution_to_clusters) - np.sum(decayed_distribution_to_clusters[top_indices])), 
            color='orange')
    plt.title('Distribution of imagenet label ' + imagenet_label_id + ' in the clusters')
    plt.legend(['All', 'Decayed'])
    plt.show()    


# %%

def find_matching_labels_and_clusters(cluster_assignment: np.ndarray, 
                                      imagenet_assignment: np.ndarray,
                                      decayed_indices: list,
                                      imagenet_element_count_threshold : int = 1000,
                                      imagenet_percentage_in_cluster_threshold : float = 0.5,
                                      cluster_percentage_in_imagenet_threshold : float = 0.4,
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
    """
    
    imagenet_cluster_counts, decayed_imagenet_cluster_counts = get_imagenet_cluster_counts_and_decayed(cluster_assignment, imagenet_assignment, decayed_indices)

    distribution_of_imagenet_labels_to_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts)
    percentages_of_imagenet_labels_in_clusters = get_distribution_of_imagenet_labels_to_clusters(imagenet_cluster_counts.T)

    im_element_count = np.sum(imagenet_cluster_counts, axis=1)
    cl_element_count = np.sum(imagenet_cluster_counts, axis=0)

    poss = np.column_stack(np.where(distribution_of_imagenet_labels_to_clusters > imagenet_percentage_in_cluster_threshold))
    rows = poss[:,0]
    cols = poss[:,1]

    relevant_labels = []
    relevant_clusters = []

    for row,col in zip(rows, cols):
        if (im_element_count[row] >= imagenet_element_count_threshold) & (percentages_of_imagenet_labels_in_clusters[col, row] >= cluster_percentage_in_imagenet_threshold):
            relevant_labels.append(row)
            relevant_clusters.append(col)

    for i in range(len(relevant_clusters)):
        plot_cluster_make_up(relevant_clusters[i], cluster_assignment, imagenet_assignment, decayed_indices, order="number")
        plot_imagenet_make_up(imagenet_cluster_counts[relevant_labels[i],:], decayed_imagenet_cluster_counts[relevant_labels[i],:], title=str(relevant_labels[i]))
        if summary:
            print("label: ", relevant_labels[i], " cluster: ", relevant_clusters[i])
            print("label in cluster: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                  " total label: ", im_element_count[relevant_labels[i]],
                  " label percentage in cluster: ", distribution_of_imagenet_labels_to_clusters[relevant_labels[i],relevant_clusters[i]])
            print("cluster in label: ", imagenet_cluster_counts[relevant_labels[i],relevant_clusters[i]],
                  " total cluster: ", cl_element_count[relevant_clusters[i]],
                  " cluster percentage in label: ", percentages_of_imagenet_labels_in_clusters[relevant_clusters[i],relevant_labels[i]])
        


# %%

find_matching_labels_and_clusters(cluster_assignment, imagenet_assignment, decayed_indices)

# %%

relevant_label = 20
plt.bar(np.arange(number_of_clusters), imagenet_cluster_counts[relevant_label,:])
plt.title('Distribution of imagenet label ' + str(relevant_label) + ' in the clusters')
plt.show()

#%%
a = []
len(a)
# %%
a = [1,2,3]
a.append(4)
a
# %%
a = np.zeros(5)
np.append(a, 1)

# %%
