from get_embeddings import main as get_embeddings_main
from get_clusters import main as get_clusters_main
from vector_search_nearest_neighbour_self_cluster import main as vector_search_main
import argparse

def main(args):
    get_embeddings_main(args)
    get_clusters_main(args)
    vector_search_main(args)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()
    main(args)