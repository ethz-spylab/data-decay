### TODOs purpose of the repo
- Some web-scale datasets (like CC3M / LAION) are provided as a list of URLs. 
This kind of dataset can *decay* over time, as the URLs become invalid or stop pointing to the original content, for whatever reason.  This 
leaves users (e.g. model developers) with a dataset that has varying degrees of coverage over different concepts covered by the dataset.

*NAME* is a tool to understand how much the dataset has decayed, and pinpoint what concepts are most affected.
The input is just a list of URLs and their captions, + a list of indices that are no longer valid.
The user can specify an embedding model (e.g. clip-ViT-L-14) or provide pre-generated embeddings.

### TODOs installation
- fill requirements.txt
- describe some hardware requirements (if embeddings are to be generated)
- describe how to install dependencies that aren't in pip, if any

### TODOs technical descriptions:
- input format for caption_url file
- input format for the list of decayed indices
- rerunning for the embeddings / etc.
- providing different embeddings (even non-text ones)

### TODOs interpreting results:
- give one example of a group + corresponding output
- group of captions
- isolation coefficient and why we care
- core/peripheral elements in the cluster
- report where the data is saved

### TODOs auto-description of groups
For this, you will need an OpenAI account and key.
- how to run the file (also OpenAI keys)
- costs for CC3M. (it was about 300k tokens on gpt-3.5-turbo-0613 -> 0.45$), but to be on the conservative side, we can upper bound by like 5$, if there were 1000 groups as is the default.)

### TODOs license
- choose MIT license
- add a LICENSE file.
- say primarily developed by Ozgur with help from Daniel/Florian. contributors welcome.