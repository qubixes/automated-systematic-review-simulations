#!/usr/bin/env python

import pickle
import sys

import pandas as pd


if __name__ == "__main__":
    trials_fp = sys.argv[1]
    hyper_choices = {}
    with open(trials_fp, "rb") as fp:
        trials = pickle.load(fp)
    print(type(trials))
    if isinstance(trials, tuple):
        print("Unpacking...")
        trials, hyper_choices = trials

    values = {key[4:]: value for key, value in trials.vals.items()}
    hyper_choices = {key[4:]: value for key, value in hyper_choices.items()}
    for key in values:
        if key in hyper_choices:
            for i in range(len(values[key])):
                values[key][i] = hyper_choices[key][values[key][i]]

    values.update({"loss": trials.losses()})
    pd.options.display.max_rows = 999
    pd.options.display.width = 0
    print(pd.DataFrame(values).sort_values("loss"))
