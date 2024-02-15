import numpy as np
import os
from collections import Counter
import argparse

from tqdm import tqdm

from utils import get_top_n_indices, load_json, save_json, load_captions, load_yaml_munch


def main(args):
    
    args = load_yaml_munch(args.config_file)

    captions = load_captions(args.captions_urls_path)

    # Load the dataset embeddings
    if args.verbose:
        print("Loading dataset embeddings")
    dataset_embeddings_path = args.dataset_embeddings_path
    dataset_embeddings = np.load(dataset_embeddings_path)

    dataset_size = dataset_embeddings.shape[0]
    if args.verbose:
        print(f'Number of dataset samples: {dataset_size}')

    # Load the list of decayed indices
    decayed_indices = load_json(args.decayed_indices_path)

    decayed_indices_size = len(decayed_indices)
    if args.verbose:
        print(f'Number of decayed indices: {decayed_indices_size}')

    decayed_array = np.zeros(dataset_size)
    decayed_array[decayed_indices] = 1

    # Load the cluster centers, distances, and similarities
    if args.verbose:
        print("Loading cluster centers.")
    cluster_centers_path = os.path.join(args.clusters_folder, 'cluster_centers.npy')
    cluster_centers = np.load(cluster_centers_path)

    if args.similarity_type == 'distances':
        if args.verbose:
            print("Loading distances")
        distances_path = os.path.join(args.clusters_folder, 'distances.npy')
        distances = np.load(distances_path)
        similarity = 1 - distances
    elif args.similarity_type == 'dot_products':
        if args.verbose:
            print("Loading similarities")
        dot_products_path = os.path.join(args.clusters_folder, 'dot_products.npy')
        dot_products = np.load(dot_products_path)
        similarity = dot_products
    else:
        raise ValueError("Similarity type should be either distances or dot_products.")
    # Find the cluster assignments using argmax
    cluster_assignment = np.argmax(similarity, axis=1)

    
    cluster_counts = len(cluster_centers)
    decayed_dict = {}
    for i in range(len(decayed_indices)):
        decayed_dict[decayed_indices[i]] = i
        
    
    # For each decayed sample, find the closest n samples (decayed or not) in the same cluster

    if os.path.exists(args.decayed_samples_dict_nn_path) and not args.decayed_dict_calculate:
        print(f'Loading decayed samples dict nn from {args.decayed_samples_dict_nn_path}')
        diclist_nn = load_json(args.decayed_samples_dict_nn_path)
    else:
        print(f'Creating decayed samples dict nn at {args.decayed_samples_dict_nn_path}')

        # For each cluster find the samples assigned to it
        clusters_all = []
        clusters_decayed = []
        for i in range(cluster_counts):
            clusters_all.append(np.where((cluster_assignment == i))[0])
            clusters_decayed.append(np.where((cluster_assignment == i) & (decayed_array == 1))[0])

        nearby_sample_count = args.nearby_sample_count
        # For each decayed sample, find the closest n samples (decayed or not) 
        diclist_nn = [{'decayed_indice':x,
                    'nn_indices':None,
                    'nn_scores':None,
                    'nn_decayed_count':None} for x in decayed_indices]

        # Realize that similar_inds = get_top_n_indices(temp[j], nearby_sample_count) will include
        # the decayed samples as well. So we add 1 to the nearby_sample_count and then remove the first
        for i in tqdm(range(cluster_counts)):
            temp = dataset_embeddings[clusters_decayed[i]] @ dataset_embeddings[clusters_all[i]].T
            for j in range(len(clusters_decayed[i])):
                decayed_ind = clusters_decayed[i][j]
                similar_inds = get_top_n_indices(temp[j], nearby_sample_count + 1)
                nn_decayed_inds = clusters_all[i][similar_inds]

                similar_inds = np.delete(similar_inds, np.where(nn_decayed_inds == decayed_ind))
                nn_decayed_inds = np.delete(nn_decayed_inds, np.where(nn_decayed_inds == decayed_ind))
                
                diclist_nn[decayed_dict[decayed_ind]]['nn_indices'] = nn_decayed_inds.tolist()
                diclist_nn[decayed_dict[decayed_ind]]['nn_decayed_count'] = int(decayed_array[nn_decayed_inds].sum())
                diclist_nn[decayed_dict[decayed_ind]]['nn_scores'] = temp[j][similar_inds].tolist()

        save_json(args.decayed_samples_dict_nn_path, diclist_nn)

    
    # create a dict for diclist_nn
    """ diclist_nn_dict = {}
    for i in range(len(diclist_nn)):
        diclist_nn_dict[diclist_nn[i]['decayed_indice']] = i """

    
    # now include the closest clusters

    nn_decayed_counts = [diclist_nn[x]['nn_decayed_count'] for x in range(len(diclist_nn))]
    check = np.array(nn_decayed_counts) >= args.nearby_decayed_sample_count_threshold - args.closest_cluster_check_threshold
    decayed_of_interest = np.where(check)[0]
    decayed_of_interest = [diclist_nn[x]['decayed_indice'] for x in decayed_of_interest]
    decayed_of_interest = np.array(decayed_of_interest)

    if args.verbose:
        print(f'Start looking at closest {args.closest_clusters_count} clusters to the decayed samples of interest')

    diclist_nn_close_k = [{'decayed_indice':x,
                    'nn_indices_close_k':diclist_nn[decayed_dict[x]]['nn_indices'].copy(),
                    'nn_scores_close_k':diclist_nn[decayed_dict[x]]['nn_scores'].copy(),
                    'nn_decayed_count_close_k':None} for x in decayed_of_interest]

    decayed_interest_dict = {}
    for i in range(len(decayed_of_interest)):
        decayed_interest_dict[decayed_of_interest[i]] = i

    # Find the top_k clusters for each decayed sample of interest
    close_k = args.closest_clusters_count
    decayed_of_interest_clusters = []
    for i in range(len(decayed_of_interest)):
        decayed_ind = decayed_of_interest[i]
        decayed_of_interest_clusters.append(get_top_n_indices(similarity[decayed_ind],close_k+1)[1:])

    # For each cluster, find the decayed samples(top_k) in that cluster
    clusters_decayed_of_interest = [[] for _ in range(cluster_counts)]
    for i in range(len(decayed_of_interest)):
        for j in range(close_k):
            clusters_decayed_of_interest[decayed_of_interest_clusters[i][j]].append(decayed_of_interest[i])

    # For each cluster find the samples assigned to it
    clusters_all = []
    for i in range(cluster_counts):
        clusters_all.append(np.where((cluster_assignment == i))[0])

    # For each decayed sample, find the closest n samples (decayed or not)
    nearby_sample_count = args.nearby_sample_count
    for i in tqdm(range(cluster_counts)):
        temp = dataset_embeddings[clusters_decayed_of_interest[i]] @ dataset_embeddings[clusters_all[i]].T

        # we no longer need to remove the first element since our decayed samples are not included
        for j in range(len(clusters_decayed_of_interest[i])):
            decayed_ind = clusters_decayed_of_interest[i][j]
            similar_inds = get_top_n_indices(temp[j], nearby_sample_count)
            nn_decayed_inds = clusters_all[i][similar_inds]
            diclist_nn_close_k[decayed_interest_dict[decayed_ind]]['nn_indices_close_k'].extend(nn_decayed_inds.tolist())
            diclist_nn_close_k[decayed_interest_dict[decayed_ind]]['nn_scores_close_k'].extend(temp[j][similar_inds].tolist())

    # For each decayed sample of interest, we have (top_k + 1)*nearby_sample_count similar samples.
    # find top nearby_sample_count among them
    for i in tqdm(range(len(decayed_of_interest))):
        nn_indices_close_k = np.array(diclist_nn_close_k[i]['nn_indices_close_k'])
        nn_scores_close_k = np.array(diclist_nn_close_k[i]['nn_scores_close_k'])
        temp_inds = get_top_n_indices(nn_scores_close_k, nearby_sample_count)
        nn_temp_inds = nn_indices_close_k[temp_inds]
        diclist_nn_close_k[i]['nn_indices_close_k'] = nn_temp_inds.tolist()
        diclist_nn_close_k[i]['nn_scores_close_k'] = nn_scores_close_k[temp_inds].tolist()
        diclist_nn_close_k[i]['nn_decayed_count_close_k'] = int(decayed_array[nn_temp_inds].sum())

    nn_decayed_counts = [diclist_nn_close_k[x]['nn_decayed_count_close_k'] for x in range(len(diclist_nn_close_k))]

    
    # Apply thresholds for number of decayed samples nearby and minimum similarity
    check = np.array(nn_decayed_counts) >= args.nearby_decayed_sample_count_threshold

    if args.check_similarity:
        nn_score_th = args.lower_similarity_threshold
        nn_scores_close_k = [diclist_nn_close_k[x]['nn_scores_close_k'] for x in range(len(diclist_nn_close_k))]
        nn_indices_close_k = [diclist_nn_close_k[x]['nn_indices_close_k'] for x in range(len(diclist_nn_close_k))]

        check2 = []
        for i in range(len(nn_scores_close_k)):
            nn_decayed_inds_temp = decayed_array[np.array(nn_indices_close_k[i])]==1
            nn_decayed_scores_temp = np.array(nn_scores_close_k[i])[nn_decayed_inds_temp]
            if len(nn_decayed_scores_temp) < args.nearby_decayed_sample_count_threshold:
                check2.append(False)
            else:
                check2.append(nn_decayed_scores_temp[args.nearby_decayed_sample_count_threshold - 1] >= nn_score_th)
        check = check & check2

    good_ones = np.where(check)[0]
    orig_good_indices = [diclist_nn_close_k[x]['decayed_indice'] for x in good_ones]

    orig_good_indices_dict = {}
    for i in range(len(orig_good_indices)):
        orig_good_indices_dict[orig_good_indices[i]] = i

    if args.verbose:
        print(f'Number of good ones: {len(good_ones)}')
        print(f'Number of good indices, pre-filter for decayed: {len(orig_good_indices)}')

    good_indices = orig_good_indices.copy()
    if args.consider_nns:
        if args.verbose:
            print("Considering the neighbours")
        good_indices_neighbours = []
        for i in good_ones:
            good_indices_neighbours.extend(diclist_nn_close_k[i]['nn_indices_close_k'])
        good_indices.extend(good_indices_neighbours)
        good_indices = np.unique(good_indices).tolist()

    # remove the non-decayed samples from the good_indices
    good_indices = np.array(good_indices)
    good_indices = good_indices[decayed_array[good_indices]==1].tolist()

    if args.verbose:
        print(f'Number of good indices, post-filter for decayed: {len(good_indices)}')
    
    diclist_orig_good_ones = [diclist_nn_close_k[x].copy() for x in good_ones]
    
    for i in range(len(diclist_orig_good_ones)):
        diclist_orig_good_ones[i]['decayed_nn_indices'] = [x for x in diclist_orig_good_ones[i]['nn_indices_close_k'] if decayed_array[x]==1]
    
    good_indices_dict = {}
    for i in range(len(good_indices)):
        good_indices_dict[good_indices[i]] = i

    
    diclist_good_ones = {}
    for good_indice in good_indices:
        diclist_good_ones[good_indice] = []

    for i, orig_good_indice in enumerate(orig_good_indices):
        diclist_good_ones[orig_good_indice] = diclist_orig_good_ones[i]['decayed_nn_indices']
        to_dos = diclist_orig_good_ones[i]['decayed_nn_indices']
        if args.consider_nns:
            for to_do in to_dos:
                if to_do not in orig_good_indices_dict:
                    diclist_good_ones[to_do].append(orig_good_indice)

    
    # orig_good_indices are the decayed samples that fulfill the conditions
    # good_indices are the decayed samples that fulfill the conditions and their decayed neighbours
    # good_indices_dict is a dictionary that maps the good_indices to their indices in good_indices
    # diclist_good_ones is a dictionary that maps the good_indices to their decayed neighbours
    # diclist_orig_good_ones is the original dic_list for orig_good_indices
    # orig_good_indices_dict is a dictionary that maps the orig_good_indices to their indices in orig_good_indices
    
    # The problem is sometimes while one sample is in the nn_indices_close_k of another, the latter is not in the nn_indices_close_k of the former
    groups = np.ones(len(good_indices), dtype=int)*-1
    groups_counter = 0
    for i in range(len(good_indices)):
        if groups[i] != -1:
            continue
        groups[i] = groups_counter
        to_dos = []
        to_dos.extend(diclist_good_ones[good_indices[i]])
        while len(to_dos) > 0:
            temp_indice = to_dos.pop()
            if temp_indice in good_indices_dict:
                temp_indice_pos = good_indices_dict[temp_indice]
                if groups[temp_indice_pos] == -1:
                    groups[temp_indice_pos] = groups_counter
                    to_dos.extend(diclist_good_ones[temp_indice])
                elif groups[temp_indice_pos] != groups_counter:
                    old_cluster_count = groups[temp_indice_pos]
                    groups[groups==old_cluster_count] = groups_counter
        
        groups_counter += 1
    
    # find the group centers and combine the group that are too close to each other
    # unique_group is already sorted
    # find the cosine similarity between the group centers. put 0 for the diagonal
    # put 0's for upper triangular part

    good_dataset_embeddings = dataset_embeddings[np.array(good_indices)]

    unique_groups = np.unique(groups)
    num_groups_new = len(unique_groups)
    group_centers = np.zeros((num_groups_new, good_dataset_embeddings.shape[1]))
    for i in range(num_groups_new):
        group_centers[i] = np.average(good_dataset_embeddings[groups == unique_groups[i]], axis=0)
        group_centers[i] = group_centers[i]/np.linalg.norm(group_centers[i])

    group_similarity = group_centers @ group_centers.T
    np.fill_diagonal(group_similarity, 0)
    group_similarity = np.tril(group_similarity)

    group_similartity_threshold = args.group_similartity_threshold
    rows, cols = np.where(group_similarity > group_similartity_threshold)

    while len(rows) > 0:

        for i in range(num_groups_new - 1, 0, -1):
            temp_sim = group_similarity[i,:]
            am = np.argmax(temp_sim)
            if temp_sim[am] > group_similartity_threshold:
                groups[groups==unique_groups[i]] = unique_groups[am]

        unique_groups = np.unique(groups)
        num_groups_new = len(unique_groups)
        group_centers = np.zeros((num_groups_new, good_dataset_embeddings.shape[1]))
        for i in range(num_groups_new):
            group_centers[i] = np.average(good_dataset_embeddings[groups == unique_groups[i]], axis=0)
            group_centers[i] = group_centers[i]/np.linalg.norm(group_centers[i])

        group_similarity =group_centers @ group_centers.T
        np.fill_diagonal(group_similarity, 0)
        group_similarity = np.tril(group_similarity)

        rows, cols = np.where(group_similarity > group_similartity_threshold)

    unique_groups = np.unique(groups)
    num_groups_new = len(unique_groups)
    if args.verbose:
        print(f'Number of groups: {num_groups_new}')
    
    counter = Counter(groups)
    
    # look at groups with # elements > self.groups_element_threshold
    group_element_threshold = args.group_element_threshold
    relevant_groups = [x[0] for x in counter.items() if x[1] > group_element_threshold]
    
    # find the captions of the good_indices in those groups
    final_groups_indices = [np.array(good_indices)[groups==x].tolist() for x in relevant_groups]
    final_groups_captions = [captions[x].tolist() for x in final_groups_indices]

    if args.verbose:
        print(f'Number of groups with more than {group_element_threshold} elements: {len(relevant_groups)}')
        print(f'Number of good_indices: {len(good_indices)}')
    
    # Find the average similarity captions in a group to that group's center
    average_similarities = [[] for _ in range(len(relevant_groups))]
    average_decayed_neighbours = [[] for _ in range(len(relevant_groups))]
    neighbours_count = [[] for _ in range(len(relevant_groups))]

    if args.consider_nns:
        for i in range(len(relevant_groups)):
            group_caption_embeddings = dataset_embeddings[final_groups_indices[i]]
            group_center_embedding = np.average(group_caption_embeddings, axis=0)
            group_center_embedding /= np.linalg.norm(group_center_embedding)
            within_group_sim = group_caption_embeddings @ group_center_embedding
            average_similarities[i] = np.average(within_group_sim)

            decayed_neighbour_count = 0
            neighbourhood_count = 0
            for decayed_ind in final_groups_indices[i]:
                if decayed_ind in orig_good_indices_dict:
                    decayed_neighbour_count += diclist_nn_close_k[decayed_interest_dict[decayed_ind]]['nn_decayed_count_close_k']
                else:
                    neighbourhood_count += 1

            average_decayed_neighbours[i] = decayed_neighbour_count / (len(final_groups_indices[i]) - neighbourhood_count)
            neighbours_count[i] = neighbourhood_count
    else:
        for i in range(len(relevant_groups)):
            group_caption_embeddings = dataset_embeddings[final_groups_indices[i]]
            group_center_embedding = np.average(group_caption_embeddings, axis=0)
            group_center_embedding /= np.linalg.norm(group_center_embedding)
            within_group_sim = group_caption_embeddings @ group_center_embedding
            average_similarities[i] = np.average(within_group_sim)

            decayed_neighbour_count = 0
            for decayed_ind in final_groups_indices[i]:
                decayed_neighbour_count += diclist_nn_close_k[decayed_interest_dict[decayed_ind]]['nn_decayed_count_close_k']
            
            average_decayed_neighbours[i] = decayed_neighbour_count / len(final_groups_indices[i])
    
    if args.verbose:
        if args.consider_nns:
            for i, x in enumerate(final_groups_captions):
                print(f'group {i}, # captions: {len(x)}')
                print(f'{len(x)-neighbours_count[i]} fulfilled conditions, {neighbours_count[i]} are decayed neighbours of them')
                print(f'Average cosine similarity to group center: {average_similarities[i]:.3f}')
                print(f'Isolation factor {average_decayed_neighbours[i]:.2f}/{args.nearby_sample_count}')
                print(x[:10])
                print('\n')
        else:
            for i, x in enumerate(final_groups_captions):
                print(f'group {i}, # captions: {len(x)}')
                print(f'Average cosine similarity to group center: {average_similarities[i]:.3f}')
                print(f'Isolation factor {average_decayed_neighbours[i]:.2f}/{args.nearby_sample_count}')
                print(x[:10])
                print('\n')
    
    # save the group captions to a json file
    final_groups_captions_path = os.path.join(args.result_folder, 'group_captions.json')
    final_groups_indices_path = os.path.join(args.result_folder, 'group_indices.json')
    similarity_to_group_center_path = os.path.join(args.result_folder, 'similarity_to_group_center.npy')
    isolation_factor_path = os.path.join(args.result_folder, 'isolation_factor.npy')

    save_json(final_groups_captions_path, final_groups_captions)
    save_json(final_groups_indices_path, final_groups_indices)
    np.save(similarity_to_group_center_path, average_similarities)
    np.save(isolation_factor_path, average_decayed_neighbours)
    
    print('Completed vector search!')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)