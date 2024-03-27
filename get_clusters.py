from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pandas as pd
import os
import argparse

from utils import dot_products_distances, load_yaml_munch

def main(args):

    args = load_yaml_munch(args.config_file)

    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    # Create and check the save files
    save_folder = args.clusters_folder
    cluster_centers_path = os.path.join(save_folder, 'cluster_centers.npy')
    dot_products_path = os.path.join(save_folder, 'dot_products.npy')
    distances_path = os.path.join(save_folder, 'distances.npy')

    # Check if cluster_centers_path, dot_products_path, and distances_path exist
    if os.path.exists(cluster_centers_path) and os.path.exists(dot_products_path) and os.path.exists(distances_path):
        print(f'Files {cluster_centers_path}, {dot_products_path}, and {distances_path} already exist. Exiting')
        exit()

    if not os.path.exists(save_folder):
        print(f'Creating folder {save_folder}')
        os.makedirs(save_folder)

    # Load the embeddings
    dataset_embeddings_path = args.dataset_embeddings_path
    embeddings = np.load(dataset_embeddings_path)

    # Find the clusters
    cluster_count = args.cluster_count
    verbose = args.verbose
    if args.use_torch_kmeans:
        from kmean_torch import kmeans_core
        #km = kmeans_core(k=K,data_array=A,batch_size=3000,epochs=200)
        km = kmeans_core(k=cluster_count, data_array=embeddings, batch_size=256 * 16, epochs=1, all_cuda=True, verbose=False)
        print("Starting torch k-means")
        km.run()
        cluster_centers = km.cent.cpu().numpy()

    else:
        kmeans_fitter =  MiniBatchKMeans(n_clusters=cluster_count, batch_size=256 * 16, verbose=verbose, n_init=5, max_iter=500, random_state=42)
        print("Starting k-means")
        kmeans = kmeans_fitter.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_

    # Save the cluster centers
    np.save(cluster_centers_path, cluster_centers)

    # Find the similarities and distances of samples to clusters centers
    print("Calculating dot products and distances")
    dot_products , distances = dot_products_distances(embeddings, cluster_centers, device_c=0)

    # Save the similarities and distances
    print("Saving dot products and distances")
    np.save(dot_products_path, dot_products)
    np.save(distances_path, distances)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)