#!/usr/bin/env python
'''
Created on 23 Apr 2019

@author: qubix
'''

import sys

from asr.utils import text_to_features
from asr.readers import read_data

filename = sys.argv[1]
file_out = sys.argv[2]

print(filename)
_, text, labels = read_data(filename)
X, word_index = text_to_features(text)

with open(file_out, "w") as f:
    for key in word_index:
        f.write(f"{key}\n")
