#!/usr/bin/env python

import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

import asreview
from asreview.unsupervised.utils import get_unsupervised_class
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.cluster import KMeans
from asreview.unsupervised.cluster import normalized_cluster_score
from os.path import basename, splitext


def quality(result_list, alpha=1):
    q = 0
    for _, rank in result_list:
        q += rank**alpha

    return (q/len(result_list))**(1/alpha)


def test_unsupervised(X, y, n_test=5):

    n_clusters = max(2, int(len(y)/200))
    all_scores = []
    for _ in range(n_test):
        kmeans_model = KMeans(n_clusters=n_clusters, n_init=1, n_jobs=4)
        prediction = kmeans_model.fit_predict(X)
        score = normalized_cluster_score(prediction, y)
        all_scores.append(score)
    return all_scores


def create_objective_func(data_fp, unsupervised_class, n_unsuper_test=3,
                          n_cluster_test=20, const_param={}):
    _, texts, y = asreview.ASReviewData.from_file(data_fp).get_data()

    def objective_func(param):
        real_param = {key[4:]: value for key, value in param.items()}
        param.update(const_param)
        scores = []
        for _ in range(n_unsuper_test):
            X = unsupervised_class(param=real_param).fit_transform(texts)
            scores.extend(test_unsupervised(X, y, n_test=n_cluster_test))
        loss = -np.average(scores)
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
    data_name = splitext(basename(data_fp))[0]
    trials_fp = os.path.join("cluster", data_name, f"trials_{unsupervised_method}.pkl")
    os.makedirs(os.path.join("cluster", data_name), exist_ok=True)
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
