#!/usr/bin/env python

import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

import asreview
from asreview.models import NBModel
from hyperopt import STATUS_OK, Trials, fmin, tpe
from asreview.models.sklearn_models import CompNBModel, SVCModel
from asreview.models.utils import get_model_class
from asreview.balance_strategies.triple_balance import double_balance
import logging


def quality(result_list, alpha=1):
    q = 0
    for _, rank in result_list:
        q += rank**alpha

    return (q/len(result_list))**(1/alpha)


def test_inactive(model, fit_kwargs, X, y):

    one_idx = np.where(y == 1)[0]
    zero_idx = np.where(y == 0)[0]

    n_zero_train = round(0.75*len(zero_idx))
    n_one_train = round(0.75*len(one_idx))
    n_train = n_zero_train+n_one_train

    results = {}
    for _ in range(100):
        train_one_idx = np.random.choice(
            one_idx, n_one_train, replace=False)
        train_zero_idx = np.random.choice(
            zero_idx, n_zero_train, replace=False)
        train_idx = np.append(train_one_idx, train_zero_idx)
        test_idx = np.delete(np.arange(len(y)), train_idx)
        X_train, y_train = double_balance(X, y, train_idx)
        model.fit(X_train, y_train, **fit_kwargs)
#         model.fit(X[train_idx], y[train_idx], **fit_kwargs)

        proba_test = model.predict_proba(X[test_idx])[:, 1]
        proba_test = [(test_idx[idx], -proba) for idx, proba in enumerate(proba_test)]
        proba_test = sorted(proba_test, key=lambda x: x[1])

#         print(proba_test)
        for position, item in enumerate(proba_test):
            idx = item[0]
            if y[idx] == 1:
                if idx not in results:
                    results[idx] = [0, 0]
                results[idx][0] += position
                results[idx][1] += 1

    result_list = []
    for key, item in results.items():
        new_value = item[0]/(item[1]*(len(y)-n_train))
        result_list.append([key, new_value])

    result_list = sorted(result_list, key=lambda x: x[1])

    return quality(result_list, 1.0)


def create_objective_func(data_fp, model_class, _const_param={}):
    _, texts, labels = asreview.ASReviewData.from_file(data_fp).get_data()

    def objective_func(param):
        real_param = {key[4:]: value for key, value in param.items()}
        wrap_model = model_class(real_param)
        model = wrap_model.model()
        fit_kwargs = wrap_model.fit_kwargs()
        X, y = wrap_model.get_Xy(texts, labels)
        loss = test_inactive(model, fit_kwargs, X, y)
        return {"loss": loss, 'status': STATUS_OK}

    return objective_func


if __name__ == "__main__":
#     logging.getLogger().setLevel(logging.DEBUG)
    data_fp = sys.argv[1]
    model_name = sys.argv[2]
    n_iter = int(sys.argv[3])
    model_class = get_model_class(model_name)

    obj_function = create_objective_func(data_fp, model_class)
    hyper_space, _ = model_class().hyper_space()
    trials_fp = os.path.join(f"trials_{model_class().name}.pkl")
    if trials_fp is not None:
        try:
            with open(trials_fp, "rb") as fp:
                trials = pickle.load(fp)
        except FileNotFoundError:
            trials = None
            print(f"Cannot find {trials_fp}")

    if trials is None:
        trials = Trials()
        n_start_evals = 0
    else:
        n_start_evals = len(trials.trials)

    for i in tqdm(range(n_iter)):
        fmin(fn=obj_function,
             space=hyper_space,
             algo=tpe.suggest,
             max_evals=i+n_start_evals+1,
             trials=trials,
             show_progressbar=False)
        with open(trials_fp, "wb") as fp:
            pickle.dump(trials, fp)
