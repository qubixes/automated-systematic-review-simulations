#!/usr/bin/env python

import pickle
import sys

import pandas as pd


if __name__ == "__main__":
    trials_fp = sys.argv[1]
    with open(trials_fp, "rb") as fp:
        trials = pickle.load(fp)

    values = {key[4:]: value for key, value in trials.vals.items()}
    values.update({"loss": trials.losses()})
    pd.options.display.max_rows = 999
    pd.options.display.width = 0
    print(pd.DataFrame(values).sort_values("loss"))
