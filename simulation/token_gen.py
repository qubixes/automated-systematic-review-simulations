#!/usr/bin/env python

import sys

import asreview


filename = sys.argv[1]
file_out = sys.argv[2]

_, text, labels = asreview.ASReviewData.from_file(filename).get_data()
_, word_index = asreview.text_to_features(text)

with open(file_out, "w") as f:
    for key in word_index:
        f.write(f"{key}\n")
