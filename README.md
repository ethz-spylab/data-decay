# *DecayedConcepts*

*DecayedConcepts* is a tool for analyzing the decay of web-scale datasets provided as a list of URLs, such as CC3M or LAION. Over time, these datasets can deteriorate as URLs become invalid or stop pointing to the original content, resulting in varying degrees of coverage over different concepts. *DecayedConcepts* helps understand the extent of the decay and pinpoints the most affected concepts.

## Features

- Analyzes the decay of web-scale datasets given as a list of URLs and captions
- Identifies decayed patches in the embedding space and returns the captions of the decayed samples
- Supports user-specified embedding models (e.g., clip-ViT-L-14) or pre-generated embeddings
- Provides detailed patch reports with information on core and peripheral samples, isolation coefficient, and average cosine similarity
- Enables auto-description of groups using GPT for concept generation

## Requirements

To set up the environment for DecayedConcepts, follow these steps:

```bash
conda create -n decayedvenv python=3.9
conda activate decayedvenv
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the CC3M caption-url format. For other formats, update the `load_captions` function in `utils.py`.
2. Provide the decayed indices as a JSON file. For other formats, update the relevant line in `find_patches.py`.
3. Run `decayed_concepts.py` with the `config.yml` file to perform embedding calculation, clustering, and decayed concept search.

```bash
python decayed_concepts.py config.yml
```
This is equivalent to 
```bash
python get_embeddings.py config.yml
python get_clusters.py config.yml
python find_patches.py config.yml
```

Note:
- Embedding calculation is a one-time process. To repeat it, delete the embeddings file and run the program again.
- Clustering can be repeated with different numbers of clusters by deleting the current cluster files and running the code with the new `cluster_count` parameter.
- Vector search can be repeated multiple times with different hyperparameters. The `decayed_dict` re-calculation is only necessary if the `similarity_type` or `nearby_sample_count` changes.

## Interpreting Results

DecayedConcepts aims to find decayed concepts by assuming that samples belonging to a concept are clustered together in the embedding space. The tool searches for isolated patches of decayed samples and provides detailed patch reports.

The patch reports include:
- Number of samples in the patch and the count of core and peripheral samples
- Average cosine similarity to the patch center
- Isolation coefficient of core samples (indicating how isolated the patch is from non-decayed samples)
- Captions of the samples in the patch

An example output:

```
['group 4, # captions: 518',
  '390 core decayed captions in the group, 128 peripheral decayed captions',
  'Average cosine similarity to group center: 0.816',
  'Isolation coefficient of core decayed captions: 0.89 on neighborhood sample 20',
  '"family picnic with a dog" * 19, "happy family in the park" * 13, "the family is resting on the lawn in the city park ." * 10 ...]
```

Peripheral samples can be excluded by setting `consider_nns` to `false` in `config.yml`.

## Auto-Description of Groups using GPT

DecayedConcepts supports auto-description of groups using GPT for concept generation. To use this feature:

1. Set up an OpenAI account and obtain an API key.
2. Modify `gpt_concept_generation.py` for prompt-tuning.
3. Run `gpt_concept_generation.py` to generate concepts for each patch.

The generated concepts will be printed and saved in the `generated_captions.json` file.

## License
MIT License

Copyright (c) 2024 Ozgur Celik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
