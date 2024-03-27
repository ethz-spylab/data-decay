# %%
from openai import OpenAI
import dotenv
from pathlib import Path
import random
import json
from pathlib import Path


generated_captions_path_dir = Path('/data/projects/data-decay/cc3m/script_tests/results/')


dotenv.load_dotenv()

client = OpenAI()

# %%

cluster_captions_path = Path(f'/data/projects/data-decay/cc3m/script_tests/results/group_captions.json')
with open(cluster_captions_path, "r") as f:
    cluster_captions = json.load(f)

print(f"Number of clusters: {len(cluster_captions)}")

results = []
for i,relevant_captions in enumerate(cluster_captions):
    if len(relevant_captions) == 0:
        print(f"\nCluster {i+1} is empty.")
        continue
    print(f"\nCluster {i}:")
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

# %%    
