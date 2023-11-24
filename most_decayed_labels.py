import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_top_n_indices
import json
import pickle
import tqdm

parser = argparse.ArgumentParser(description='Find the most decayed labels. The dataset sample to label assignments are done with similarity thresholding. \
                                    This means a sample might zero, one, or more labels.')

parser.add_argument('--dataset_embeddings_path', type=str, default='embeddings/text_embeddings.npy',
                    help='embeddings path for the dataset. This can be either the image or caption embeddings.')
parser.add_argument('--labels_path', type=str, default='labels.txt',
                    help='labels path')
parser.add_argument('--label_embeddings_path', type=str, default='embeddings/labels_embeddings.npy',
                    help='label embeddings path')
parser.add_argument('--decayed_indices_path', type=str, default='decayed_indices.txt',
                    help='decayed indices path')
parser.add_argument('--assignments_folder', type=str, default='assignments/',
                    help='assignments folder will include the assignments from labels to dataset samples, and dataset samples to labels. If it does not exist, it will be created.')
parser.add_argument('--similarity_threshold', type=float, default=0.5,
                    help='similarity threshold for assigning a sample to a label')
parser.add_argument('--display_similarity_threshold', type=float, default=0.5,
                    help='similarity threshold for displaying a label vs dataset sample plot')
parser.add_argument('--sample_count_threshold', type=int, default=1000,
                    help='how many samples a label should be present in the dataset to be considered')
parser.add_argument('--display_plots_count', type=int, default=200,
                    help='how many plots to display')
parser.add_argument('--verbose', type=int, default=1,
                    help='verbosity of code')

args = parser.parse_args()

# Read the labels
labels_path = args.labels_path
labels = []
with open(labels_path, "r") as f:
    for line in f:
        labels.append(line.strip())

labels_size = len(labels)

# Load the list of decayed indices
decayed_indices_path = args.decayed_indices_path
decayed_indices = []
with open(decayed_indices_path, "r") as f:
    for line in f:
        decayed_indices.append(int(line.strip()))


# Load the dataset embeddings
if args.verbose:
    print("Loading dataset embeddings")
dataset_embeddings_path = args.dataset_embeddings_path
dataset_embeddings = np.load(dataset_embeddings_path)

dataset_size = dataset_embeddings.shape[0]

# Create the decayed array
decayed_array = np.zeros(dataset_size)
decayed_array[decayed_indices] = 1

# Load the label embeddings
if args.verbose:
    print("Loading label embeddings")
label_embeddings_path = args.label_embeddings_path
label_embeddings = np.load(label_embeddings_path)

# Find dataset and label embeddings comparison
assert dataset_embeddings.shape[1] == label_embeddings.shape[1], "Dataset and label embeddings should have the same dimension."
dataset_vs_label = np.matmul(dataset_embeddings, label_embeddings.T)

# Find the assignments
if not os.path.exists(args.assignments_folder):
    print(f'Creating folder {args.assignments_folder}')
    os.makedirs(args.assignments_folder)

assignments_folder = args.assignments_folder

label_to_dataset_assignments_path = os.path.join(assignments_folder, 'label_to_dataset_assignments.txt')
if os.path.exists(label_to_dataset_assignments_path):
    print(f'Loading label to dataset assignments from {label_to_dataset_assignments_path}')
    with open(label_to_dataset_assignments_path) as fp:
        label_to_dataset = json.load(fp)
    label_to_dataset = [np.array(label_to_dataset[i], dtype=int) for i in range(labels_size)]
else:
    print(f'Creating label to dataset assignments at {label_to_dataset_assignments_path}')
    label_to_dataset = []
    for i in tqdm(range(labels_size)):
        label_to_dataset.append(np.where(dataset_vs_label[:, i] > args.similarity_threshold)[0])
    label_to_dataset2 = [label_to_dataset[i].tolist() for i in range(labels_size)]
    with open(label_to_dataset_assignments_path, 'w') as fp:
        json.dump(label_to_dataset2, fp)
    del label_to_dataset2

dataset_to_label_assignments_path = os.path.join(assignments_folder, 'dataset_to_label_assignments.txt')
if os.path.exists(dataset_to_label_assignments_path):
    print(f'Loading dataset to label assignments from {dataset_to_label_assignments_path}')
    with open(dataset_to_label_assignments_path) as fp:
        dataset_to_label = json.load(fp)
    dataset_to_label = [np.array(dataset_to_label[i], dtype=int) for i in range(labels_size)]
else:
    print(f'Creating dataset to label assignments at {dataset_to_label_assignments_path}')
    dataset_to_label = []
    for i in tqdm(range(dataset_size)):
        dataset_to_label.append(np.where(dataset_vs_label[i, :] > args.similarity_threshold)[0])
    dataset_to_label2 = [dataset_to_label[i].tolist() for i in range(labels_size)]
    with open(dataset_to_label_assignments_path, 'w') as fp:
        json.dump(dataset_to_label2, fp)
    del dataset_to_label2

decayed_dataset_to_label = [dataset_to_label[i] for i in decayed_indices]

decayed_label_to_dataset_path = os.path.join(assignments_folder, 'decayed_label_to_dataset.txt')
if os.path.exists(decayed_label_to_dataset_path):
    print(f'Loading decayed label to dataset assignments from {decayed_label_to_dataset_path}')
    with open(decayed_label_to_dataset_path) as fp:
        decayed_label_to_dataset = json.load(fp)
    decayed_label_to_dataset = [np.array(decayed_label_to_dataset[i], dtype=int) for i in range(len(decayed_label_to_dataset))]
else:
    print(f'Creating decayed label to dataset assignments at {decayed_label_to_dataset_path}')
    decayed_label_to_dataset = []
    decayed_dataset_vs_label = dataset_vs_label[decayed_indices]
    for i in tqdm(range(labels_size)):
        decayed_label_to_dataset.append([decayed_indices[x] for x in np.where(decayed_dataset_vs_label[:, i] > args.similarity_threshold)[0]])
    with open(decayed_label_to_dataset_path, 'w') as fp:
        json.dump(decayed_label_to_dataset, fp)

# Calculate the distributions of (decayed) dataset samples to labels
if args.verbose:
    print('Calculating the distributions of (decayed) dataset samples to labels')
dataset_to_label_distribution = np.zeros(labels_size)
decayed_dataset_to_label_distribution = np.zeros(labels_size)
for i in tqdm(range(dataset_size)):
    dataset_to_label_distribution[dataset_to_label[i]] += 1
for i in tqdm(range(dataset_size)):
    decayed_dataset_to_label_distribution[decayed_dataset_to_label[i]] += 1

percentage_decayed_in_label = decayed_dataset_to_label_distribution / dataset_to_label_distribution
percentage_decayed_in_label[np.isnan(percentage_decayed_in_label)] = 0

percentage_decayed_in_label_cp = percentage_decayed_in_label.copy()
percentage_decayed_in_label_cp[dataset_to_label_distribution < args.sample_count_threshold] = 0
highest_percentage_num_threshold = get_top_n_indices(percentage_decayed_in_label_cp, args.display_plots_count)

decayed_dataset_vs_label = dataset_vs_label[:,decayed_array==1]
existing_dataset_vs_label = dataset_vs_label[:,decayed_array==0]

# Plot the top N highest percentage decayed labels
th = args.display_similarity_threshold
for i in range(args.display_plots_count):

    txt = i
    plt.hist(decayed_dataset_vs_label[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.hist(existing_dataset_vs_label[txt], bins=np.linspace(th,1,int((1-th)*50)+1),alpha=0.6)
    plt.legend(["Decayed", "Existing"])
    plt.xlabel("Similarity score")
    plt.ylabel("Count")
    plt.title("\""+highest_percentage_num_threshold[txt]+"\""+" vs captions")
    plt.show()

    dec = np.sum(decayed_dataset_vs_label[txt] > th)
    ex = np.sum(existing_dataset_vs_label[txt] > th)
    print(f'Top {i} highest percentage decayed label: {highest_percentage_num_threshold[txt]}')
    print(f'# of decayed: {dec}, \
        \n# of total: {ex+dec}, \
        \n% of decayed: {dec/(ex+dec)*100:.2f}')