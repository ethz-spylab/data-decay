from get_embeddings import main as get_embeddings_main
from get_clusters import main as get_clusters_main
from find_patches import main as find_patches_main
import argparse

def main(args):
    get_embeddings_main(args)
    get_clusters_main(args)
    find_patches_main(args)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()
    main(args)