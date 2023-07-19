#%%
"""
Extract up to three meaningful pieces of information from an URL:
1. the domain name
2. the descriptive part of the path (the name of the file)
3. the year
"""

from urllib.parse import urlparse
import re
from datetime import datetime
from dateutil.parser import parse
import pandas as pd
def extract_info(url):
    parsed_url = urlparse(url)

    # Extract domain name
    domain_name = parsed_url.netloc

    # Extract file name
    file_name = parsed_url.path.split('/')[-1]

    # Extract year from the url. It's in [2000, 2001, ..., 2018]
    year = None
    good_years = [str(y) for y in range(2000, 2019)]
    for y in reversed(good_years):
        if y in url:
            year = y
            break

    return domain_name, file_name, year

url = 'https://example.com/2012/03/01/sample-file.html'
print(extract_info(url))


# %%
CC_CAPTIONS_DF = "/data/cc3m/cc3m_2023/Train_GCC-training.tsv"

def load_data(DF=CC_CAPTIONS_DF):
    cc_captions_df = pd.read_csv(CC_CAPTIONS_DF, sep="\t", header=None)
    cc_captions_df.columns = ["caption", "url"]
    urls = cc_captions_df["url"].tolist()
    captions = cc_captions_df["caption"].tolist()
    return cc_captions_df, urls, captions

df, urls, captions = load_data()

print(len(urls))
print(len(captions))
N_ENTRIES = len(captions)


#%%
# Clean up the filenames, remove hashes, etc.
import re

def clean_filename(filename):
    # Remove file extension
    filename_without_ext = filename.rsplit('.', 1)[0]

    # Replace non-alphanumeric characters with space
    clean_name = re.sub(r'[^a-zA-Z0-9]', ' ', filename_without_ext)

    # Remove any sequence of spaces with a single space
    clean_name = re.sub(r'\s+', ' ', clean_name)

    # Remove any leading or trailing spaces
    clean_name = clean_name.strip()

    # remove words which contain decimal characters
    clean_name = re.sub(r'\w*\d\w*', '', clean_name).strip()

    return clean_name

print(clean_filename('hubble-eagle-nebula-wide-field-04086y.jpg'))
print(clean_filename('f44d6cfe185c9eff57553e4690219b1f--outfits-for-women-vintage-outfits.jpg'))


#%%
# Extract info from URLs
N_URLS = 100
import random
random.seed(42)
# sample indices in the range [0, N_ENTRIES)
random_indices = random.sample(range(N_ENTRIES), N_URLS)
for i in random_indices:
    url = urls[i]
    domain_name, file_name, year = extract_info(url)
    print(f"{domain_name}\n {file_name}")
    clean_name = clean_filename(file_name)
    caption = captions[i]
    print(f"{clean_name}\n {caption}\n\n")




# %%
