### Purpose of the repo
- Some web-scale datasets (like CC3M / LAION) are provided as a list of URLs. 
This kind of dataset can *decay* over time, as the URLs become invalid or stop pointing to the original content, for whatever reason.  This 
leaves users (e.g. model developers) with a dataset that has varying degrees of coverage over different concepts covered by the dataset.

- *DecayedConcepts* is a tool to understand how much the dataset has decayed, and pinpoint what concepts are most affected.
The input is just a list of URLs and their captions, + a list of indices that are no longer valid.
The user can specify an embedding model (e.g. clip-ViT-L-14) or provide pre-generated embeddings.
*DecayedConcepts* searches over the embedding space to find decayed patches and then returns the captions of the decayed samples in those patches.


### Requirements
```
conda create -n decayedvenv python=3.9
conda activate decayedvenv
pip install -r requirements.txt
```


### How to use *DecayedConcepts*:
- The current method accepts CC3M caption-url format. For other formats, please update the load_captions function in *utils.py*.
- Currently, decayed indices are accepted as json file. For other formats please update the relevant line in *find_patches.py*.
- When working on a dataset, getting the embeddings is meant to be one off occasion. To repeat it, simply delete the embeddings file and run the program again.
Clustering can be repeated again with different number of clusters. To do that, simply delete the current cluster files, and run the code with the new cluster_count parameter.
Vector search, on the other hand might be repeated multiple times with different hyperparameters. But the decayed_dict re-calculation is only necessary if similarity type or nearby sample count changes.
For other hyperparameters, decayed_dict can be reused.
- get_embeddings currently works for captions from CC3M caption-url format. This format can be changed by updating load_captions function in utils.py.
To use images, simply calculate the embeddings of it using an image encoder of choice and save them as a numpy file where rows are samples and columns are the features.
After that update the dataset_embeddings_path.
- *DecayedConcepts* is run by calling decayed_concepts.py with config.yml file. Former simply calls other files for embedding calculation, clustering and decayed concept search, in this order.



### Interpreting results:
*DecayedConcepts* tries to find the decayed concepts and works under the assumption that the samples belonging to a concept are clustered together in the embedding space. Our goal is finding such groups or patches.
We would also prefer those patches to be isolated from non-decayed samples. To achieve this we need the similarities between decayed samples and all other samples. But given the sample count this would be very computationally expensive. As a solution, we cluster samples (using the computed embeddings) and assign each sample to a cluster. Then we restrict similarity calculations only to samples belonging to same cluster (we can search over other close by clusters as well, e.g. search over closest 3 clusters). We then calculate most similar nearby_sample_count samples for each decayed sample (we also call those similar samples as neigbouring samples). For a decayed sample, if at least nearby_decayed_sample_count_threshold of those most similar samples to it are decayed and have at least lower_similarity_threshold similarity with it, we consider this decayed sample a core sample. If a neighbouring decayed sample of a core sample itself is not a core sample, then it is called a peripheral sample. We say two core samples are in the same patch if they are neighbours of each other or share a common peripheral sample. We then find the center of each patch by averaging their samples, and then combine patches that has higher than group_similartity_threshold center similarity.
For each patch, we report number of samples in that patch and how many of them are core or peripheral. We calculate isolation coefficient of core samples which is the on average how many neighbours of core samples are decayed. Higher this value gets, the more isolated (far away from non-decayed samples) a patch is. We also report the average cosine similarity to patch center. We also report all the captions in a patch. We have observed that the inclusion of peripheral samples is beneficial to understand a patch, especially true for smaller patches, but if desired peripheral samples can be excluded by setting consider_nns false in *config.yml*.LALE
Here, we are providing an example patch report:

```
['group 4, # captions: 518',
  '390 core decayed captions in the group, 128 peripheral decayed captions',
  'Average cosine similarity to group center: 0.816',
  'Isolation coefficient of core decayed captions: 0.89 on neighborhood sample 20',
  '"family picnic with a dog" * 19, "happy family in the park" * 13, "the family is resting on the lawn in the city park ." * 10 ...]
```

### Auto-description of groups using gpt
- Once the patch captions are found, gpt can be used to automatically generate concepts, yet it requires significant prompt-tuning. This can be done in *gpt_concept_generation.py*. For this, you will need an OpenAI account and key. It currently prints 3 concepts for each patch, and also saves them in the *generated_captions.json* file.
- We have used gpt-3.5-turbo-0613 and for 300k tokens the cost was 0.45$.

## License
[MIT](https://choosealicense.com/licenses/mit/)
