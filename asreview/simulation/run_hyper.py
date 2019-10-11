#!/usr/bin/env python

import pandas

from parameter_opt import optimize_svm, SVM_KERNELS, BALANCE_STRATS, SVM_GAMMA
import pickle
from pandas import DataFrame


optimize_svm(trials_fp="trials.pkl", max_evals=10)

with open('trials.pkl', 'rb') as f:
    trials = pickle.load(f)

data_dict = dict(
    kernel=[],
    svm_C=[],
    class_weight=[],
    loss=[],
    svm_gamma=[],
    balance_strategy=[],
)

for trial in trials.trials:
    loss = trial['result']['loss']
    for var in trial['misc']['vals']:
        val = trial['misc']['vals'][var][0]
        if var == 'kernel':
            val = SVM_KERNELS[val]
        elif var == 'svm_gamma':
            val = SVM_GAMMA[val]
        elif var == 'balance_strategy':
            val = BALANCE_STRATS[val]
        data_dict[var].append(val)
    data_dict["loss"].append(loss)


new_dict = {}

for var in data_dict:
    if len(data_dict[var]) > 0:
        new_dict[var] = data_dict[var]

pandas.options.display.max_rows = 999
print(DataFrame(new_dict).sort_values("loss"))
