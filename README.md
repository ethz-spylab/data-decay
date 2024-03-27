### TODOs purpose of the repo
- Some web-scale datasets (like CC3M / LAION) are provided as a list of URLs. 
This kind of dataset can *decay* over time, as the URLs become invalid or stop pointing to the original content, for whatever reason.  This 
leaves users (e.g. model developers) with a dataset that has varying degrees of coverage over different concepts covered by the dataset.

*NAME* is a tool to understand how much the dataset has decayed, and pinpoint what concepts are most affected.
The input is just a list of URLs and their captions, + a list of indices that are no longer valid.
The user can specify an embedding model (e.g. clip-ViT-L-14) or provide pre-generated embeddings.
*NAME* searches over the embedding space to find decayed patches and then returns the captions of the decayed samples in those patches.


## Requirements
```
conda create -n decayedvenv python=3.9
conda activate decayedvenv
pip install -r requirements.txt
``


### TODOs technical descriptions:
- input format for caption_url file
The current method accepts CC3M caption-url format. For other formats, please update the load_captions function in utils.py
- input format for the list of decayed indices
Currently, decayed indices are accepted as json file. For other formats please update the " decayed_indices = load_json(args.decayed_indices_path) " line in vector_search.
- rerunning for the embeddings / etc.
When working on a dataset, getting the embeddings is meant to be one off occasion.
Clustering can be repeated again with different number of clusters.
Vector search, on the other hand might be repeated multiple times with different hyperparameters. But the decayed_dict calculation there is only necessary if similarity type or nearby sample count changes.
For other hyperparameters, decayed_dict can be reused.
- providing different embeddings (even non-text ones)
get_embeddings currently works for captions from CC3M caption-url format. This format can be changed by updating load_captions function in utils.py.
To use images, simply calculate the embeddings of it using an image encoder of choice and save them as a numpy file where rows are samples and columns are the features.
After that update the dataset_embeddings_path.


### TODOs interpreting results:
- give one example of a group + corresponding output
- group of captions
- isolation coefficient and why we care
- core/peripheral elements in the cluster
- report where the data is saved
*NAME* tries to find the decayed concepts and works under the assumption that the samples belonging to a concept are clustered together in the embedding space. Our goal is finding such clusters or patches.
We would also prefer those patches to be isolated from non-decayed samples. To achieve this we need the similarities between decayed samples and all other samples. But given the sample count this would be very computationally expensive. As a solution, we cluster samples (using the computed embeddings) and assign each sample to a cluster. Then we restrict similarity calculations only to samples belonging to same cluster (we can search over other close by clusters as well, e.g. search over closest 3 clusters). We then calculate most similar nearby_sample_count samples for each decayed sample (we also call those similar samples as neigbouring samples). If, at least nearby_decayed_sample_count_threshold of those most similar samples are decayed and have at least lower_similarity_threshold similarity with it, we consider this decayed sample to be a core sample. If a neighbouring decayed sample of a core sample itself is not a core sample, then it is called a peripheral sample. We say two core samples are in the same patch if they are neighbours of each other or share a common peripheral sample. We then find the center of each patch by averaging their samples, and then combine patches that has higher than group_similartity_threshold center similarity. 

### TODOs auto-description of groups
For this, you will need an OpenAI account and key.
- how to run the file (also OpenAI keys)
- costs for CC3M. (it was about 300k tokens on gpt-3.5-turbo-0613 -> 0.45$), but to be on the conservative side, we can upper bound by like 5$, if there were 1000 groups as is the default.)

### TODOs license
- choose MIT license
- add a LICENSE file.
- say primarily developed by Ozgur with help from Daniel/Florian. contributors welcome.