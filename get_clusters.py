import numpy as np
import os
import argparse

from sklearn.cluster import KMeans, MiniBatchKMeans
from utils import dot_products_distances, load_yaml_munch

def main(args):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    save_folder = args.clusters_folder
    cluster_centers_path = os.path.join(save_folder, 'cluster_centers.npy')
    dot_products_path = os.path.join(save_folder, 'dot_products.npy')
    distances_path = os.path.join(save_folder, 'distances.npy')

    if os.path.exists(cluster_centers_path) and os.path.exists(dot_products_path) and os.path.exists(distances_path):
        print(f'Files {cluster_centers_path}, {dot_products_path}, and {distances_path} already exist. Exiting')
        return

    if not os.path.exists(save_folder):
        print(f'Creating folder {save_folder}')
        os.makedirs(save_folder)

    embeddings = np.load(args.dataset_embeddings_path)

    if args.use_torch_kmeans:
        from kmean_torch import kmeans_core
        print("Starting torch k-means")
        km = kmeans_core(k=args.cluster_count, data_array=embeddings, batch_size=256 * 16, epochs=1, all_cuda=True, verbose=args.verbose)
        km.run()
        cluster_centers = km.cent.cpu().numpy()
    else:
        print("Starting k-means")
        kmeans_fitter = MiniBatchKMeans(n_clusters=args.cluster_count, batch_size=256 * 16, verbose=args.verbose, n_init=5, max_iter=500, random_state=42)
        kmeans = kmeans_fitter.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_

    np.save(cluster_centers_path, cluster_centers)

    print("Calculating dot products and distances")
    dot_products, distances = dot_products_distances(embeddings, cluster_centers, device_c=args.cuda_device)

    print("Saving dot products and distances")
    np.save(dot_products_path, dot_products)
    np.save(distances_path, distances)

DEFAULT_CONFIG = load_yaml_munch("config.yml")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", type=bool, default=True, help="Whether to print verbose output")
    p.add_argument("--cuda_device", type=int, default=0, help="CUDA device to use")
    p.add_argument("--dataset_embeddings_path", type=str, default=DEFAULT_CONFIG.dataset_embeddings_path, help="Dataset embedding location")
    p.add_argument("--clusters_folder", type=str, default=DEFAULT_CONFIG.clusters_folder, help="Clusters save folder")
    p.add_argument("--cluster_count", type=int, default=DEFAULT_CONFIG.cluster_count, help="# of clusters to create")
    p.add_argument("--use_torch_kmeans", type=bool, default=DEFAULT_CONFIG.use_torch_kmeans, help="Use torch kmeans instead of sklearn kmeans")
    args = p.parse_args()
    main(args)