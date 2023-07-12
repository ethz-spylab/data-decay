"""
Take a file where each line is a positive integer, and subtract 2 from each line.
Save the result to a new file, appending _sub2 to the original filename, before the extension.
"""

#%%
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=Path, default="../decayed_indices_2018_to_2023.txt")
args = parser.parse_args(args=[]) if hasattr(__builtins__,'__IPYTHON__') else parser.parse_args()

#%%
new_filename = args.input.stem + "_sub2" + args.input.suffix
print(new_filename)
with open(args.input, 'r') as f:
    lines = f.readlines()
    lines = [int(x.strip()) for x in lines]
    lines = [x-2 for x in lines]
    lines = [str(x) for x in lines]
    lines = "\n".join(lines)
with open(new_filename, 'w') as f2:
    f2.write(lines)






# %%
