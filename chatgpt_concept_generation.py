# %%
from openai import OpenAI
import os
import dotenv
from pathlib import Path
import random
import json
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt

captions_embedding_path = Path("/data/cc3m/cc3m_2023/embeddings/text_embeddings_L14.npy")
decayed_indices_path = Path('/data/cc3m/script_tests/decayed_indices/combined_decayed_indices.txt')
decayed_indices_path = Path('/data/cc3m/decayed_indices.json')
generated_captions_path_dir = Path('/data/cc3m/script_tests/results/')

#cluster_counts = [10, 25, 50]
cluster_counts = None

dotenv.load_dotenv()

client = OpenAI()

# %%

if cluster_counts is None:

    cluster_captions_path = Path(f'/data/cc3m/script_tests/results/group_captions.json')
    with open(cluster_captions_path, "r") as f:
        cluster_captions = json.load(f)

    print(f"Number of clusters: {len(cluster_captions)}")

    results = []
    for i,relevant_captions in enumerate(cluster_captions):
        if len(relevant_captions) == 0:
            print(f"\nCluster {i+1} is empty.")
            continue
        print(f"\nCluster {i+1}:")
        result = []
        relevant_captions_rand = relevant_captions.copy()
        random.shuffle(relevant_captions_rand)
        len_relevant_captions = len(relevant_captions_rand)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": "You are expected to understand the three most general concepts from provided captions. List that three general concepts one by one each in a seperate line. Do not enumerate or concataneta them."},
                {"role": "user", "content": 
                    'the bride and groom dancing together at the reception .'
                    'a bride and groom share a moment at this wedding'
                    'person and groom celebrate their marriage with the traditional kiss'
                    'portrait of a bride and groom greeting'
                    'a bride and groom kissing'
                    'bride stroking the groom with her hand close up'
                    'bride walking down the aisle with her father'
                    'bride and her father dancing'
                },
                {"role": "assistant", "content":
                    #'a bride and groom'
                    #'a wedding'
                    #'a bride and her father'
                    'a bride and groom\na wedding\na bride and her father'
                },
                {"role": "user", "content": 
                    f"{relevant_captions[:100]}"
                }
            ]
        )
        print(completion.choices[0].message.content)
        result.extend(completion.choices[0].message.content.split('\n'))

        results.append(result)
        print()

    generated_captions_path = generated_captions_path_dir / f"generated_captions.json"
    with open(generated_captions_path, "w") as f:
        json.dump(results, f)    

else:
    for cluster_count in cluster_counts:

        cluster_captions_path = Path(f'/data/cc3m/script_tests/results/cluster_captions_{cluster_count}.json')
        with open(cluster_captions_path, "r") as f:
            cluster_captions = json.load(f)

        print(f"Number of clusters: {len(cluster_captions)}")

        results = []
        for i,relevant_captions in enumerate(cluster_captions):
            if len(relevant_captions) == 0:
                print(f"\nCluster {i+1} is empty.")
                continue
            print(f"\nCluster {i+1}:")
            result = []
            relevant_captions_rand = relevant_captions.copy()
            random.shuffle(relevant_captions_rand)
            len_relevant_captions = len(relevant_captions_rand)

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                    messages=[
                    {"role": "system", "content": "You are expected to understand the three most general concepts from provided captions. List that three general concepts one by one each in a seperate line. Do not enumerate or concataneta them."},
                    {"role": "user", "content": 
                        'the bride and groom dancing together at the reception .'
                        'a bride and groom share a moment at this wedding'
                        'person and groom celebrate their marriage with the traditional kiss'
                        'portrait of a bride and groom greeting'
                        'a bride and groom kissing'
                        'bride stroking the groom with her hand close up'
                        'bride walking down the aisle with her father'
                        'bride and her father dancing'
                    },
                    {"role": "assistant", "content":
                        #'a bride and groom'
                        #'a wedding'
                        #'a bride and her father'
                        'a bride and groom\na wedding\na bride and her father'
                    },
                    {"role": "user", "content": 
                        f"{relevant_captions[:100]}"
                    }
                ]
            )
            print(completion.choices[0].message.content)
            result.extend(completion.choices[0].message.content.split('\n'))

            results.append(result)
            print()

        generated_captions_path = generated_captions_path_dir / f"generated_captions_{cluster_count}.json"
        with open(generated_captions_path, "w") as f:
            json.dump(results, f)


# %%    
