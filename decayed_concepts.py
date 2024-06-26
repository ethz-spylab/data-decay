from get_embeddings import main as get_embeddings_main
from get_clusters import main as get_clusters_main
from find_patches import main as find_patches_main
from utils import load_yaml_munch
import argparse
from pathlib import Path

DEFAULT_CONFIG = load_yaml_munch("config.yml")

def main():
    parser = argparse.ArgumentParser(description="Run the DecayedConcepts analysis pipeline.")
    parser.add_argument("--verbose", type=bool, default=True, help="Whether to print landmark actions and results")
    parser.add_argument("--cuda_device", type=int, default=0, help="Cuda device to use")
    parser.add_argument("--captions_urls_path", type=str, default=DEFAULT_CONFIG.captions_urls_path, help="Location of the captions and urls")
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG.model_name, help="Model to use for the embeddings")
    parser.add_argument("--step_size", type=int, default=DEFAULT_CONFIG.step_size, help="Step size for calculating embeddings")
    parser.add_argument("--dataset_embeddings_path", type=str, default=DEFAULT_CONFIG.dataset_embeddings_path, help="Dataset embedding location")
    parser.add_argument("--cluster_count", type=int, default=DEFAULT_CONFIG.cluster_count, help="# of clusters to create")
    parser.add_argument("--clusters_folder", type=str, default=DEFAULT_CONFIG.clusters_folder, help="Clusters save folder")
    parser.add_argument("--use_torch_kmeans", type=bool, default=DEFAULT_CONFIG.use_torch_kmeans, help="Use torch kmeans instead of sklearn kmeans")
    parser.add_argument("--decayed_indices_path", type=str, default=DEFAULT_CONFIG.decayed_indices_path, help="Location of decayed indices")
    parser.add_argument("--decayed_dict_calculate", type=bool, default=DEFAULT_CONFIG.decayed_dict_calculate, help="Only need to recalculate decayed samples dictionary if decayed indices or the nearby_sample_count are updated")
    parser.add_argument("--decayed_samples_dict_nn_path", type=str, default=DEFAULT_CONFIG.decayed_samples_dict_nn_path, help="Location of decayed samples dictionary")
    parser.add_argument("--consider_nns", type=bool, default=DEFAULT_CONFIG.consider_nns, help="True if we want to consider the peripheral samples")
    parser.add_argument("--similarity_type", type=str, default=DEFAULT_CONFIG.similarity_type, help="Which similarity measure to use distances or dot products")
    parser.add_argument("--result_folder", type=str, default=DEFAULT_CONFIG.result_folder, help="Where to save the results. Make sure folder exists")
    parser.add_argument("--nearby_sample_count", type=int, default=DEFAULT_CONFIG.nearby_sample_count, help="Number of nearest neighbors to consider")
    parser.add_argument("--nearby_decayed_sample_count_threshold", type=int, default=DEFAULT_CONFIG.nearby_decayed_sample_count_threshold, help="At least how many of those nearest neighbors should be decayed")
    parser.add_argument("--closest_clusters_count", type=int, default=DEFAULT_CONFIG.closest_clusters_count, help="How many other clusters to consider apart from the closest one (from closest to farthest)")
    parser.add_argument("--closest_cluster_check_threshold", type=int, default=DEFAULT_CONFIG.closest_cluster_check_threshold, help="Checking other clusters for all decayed samples might be costly. We can limit the number of decayed samples to check")
    parser.add_argument("--check_similarity", type=bool, default=DEFAULT_CONFIG.check_similarity, help="Whether to check if the nearby_decayed_sample_count_threshold decayed neighbour has at least lower_similarity_threshold similarity to the decayed sample")
    parser.add_argument("--lower_similarity_threshold", type=float, default=DEFAULT_CONFIG.lower_similarity_threshold, help="Lower similarity threshold for considering a decayed sample's neighbour")
    parser.add_argument("--group_similarity_threshold", type=float, default=DEFAULT_CONFIG.group_similarity_threshold, help="Combine clusters if their centroids have at least cluster_similarity_threshold similarity")
    parser.add_argument("--group_element_threshold", type=int, default=DEFAULT_CONFIG.group_element_threshold, help="How many decayed samples should be in a cluster to consider it (including the decayed neighbours if consider_nns is True)")

    args = parser.parse_args()

    # Pass the parsed arguments to each main function
    get_embeddings_main(args)
    get_clusters_main(args)
    find_patches_main(args)

if __name__ == "__main__":
    main()
