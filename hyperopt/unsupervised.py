#!/usr/bin/env python

import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import asreview
from asreview.unsupervised import Doc2Vec, Tfidf
from asreview.unsupervised.utils import get_unsupervised_class
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.ensemble import RandomForestClassifier


def quality(result_list, alpha=1):
    q = 0
    for _, rank in result_list:
        q += rank**alpha

    return (q/len(result_list))**(1/alpha)


def test_unsupervised(X, y):

    one_idx = np.where(y == 1)[0]
    zero_idx = np.where(y == 0)[0]

    n_zero_train = round(0.75*len(zero_idx))
    n_one_train = round(0.75*len(one_idx))
    n_train = n_zero_train+n_one_train

    results = {}
    for _ in range(10):
        train_one_idx = np.random.choice(
            one_idx, n_one_train, replace=False)
        train_zero_idx = np.random.choice(
            zero_idx, n_zero_train, replace=False)
        train_idx = np.append(train_one_idx, train_zero_idx)
        test_idx = np.delete(np.arange(len(y)), train_idx)
        model = RandomForestClassifier()
        model.fit(X[train_idx], y[train_idx])

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


def create_objective_func(data_fp, unsupervised_class, const_param={}):
    _, texts, y = asreview.ASReviewData.from_file(data_fp).get_data()

    def objective_func(param):
        real_param = {key[4:]: value for key, value in param.items()}
        param.update(const_param)
        X = unsupervised_class(param=real_param).fit_transform(texts)
        loss = test_unsupervised(X, y)
        return {"loss": loss, 'status': STATUS_OK}

    return objective_func


if __name__ == "__main__":
    data_fp = sys.argv[1]
    unsupervised_method = sys.argv[2]
    n_iter = int(sys.argv[3])
#     const_param = {"dbow_words": 0, "dm_concat": 0}

    unsupervised_class = get_unsupervised_class(unsupervised_method)
    obj_function = create_objective_func(data_fp, unsupervised_class)
    hyper_space, _ = unsupervised_class().hyper_space()
    trials_fp = os.path.join("unsupervised", f"trials_rf_{unsupervised_method}.pkl")
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
