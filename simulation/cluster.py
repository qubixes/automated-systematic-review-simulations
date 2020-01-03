#!/usr/bin/env python

from math import sqrt
import os
import sys

from asreview.readers import ASReviewData
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score
import numpy as np
from sklearn.cluster import DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from numpy import average
from scipy.stats.stats import sem


def get_all_models():
    models = {}
    for n_clusters in [2, 3, 5, 8, 15, 30]:
        models[f"kmeans_{n_clusters}"] = KMeans(
            n_clusters=n_clusters, n_init=2, n_jobs=-1)
#     models["gaussian_mix"] = GaussianMixture()
#     models["birch"] = Birch()
    return models


def simulate_score(one_dict, all_dict, n_run=10000):
    total_one = np.sum([x for x in one_dict.values()])
    total = np.sum([x for x in all_dict.values()])
    sim_scores = []
    for _ in range(n_run):
        one_idx = np.random.choice(range(total), total_one, replace=False)
        one_idx = np.sort(one_idx)
        new_one_dict = {}
        cur_all_idx = 0
        cur_one_idx = 0
#         print(one_idx)
        for key in all_dict:
            cur_all_idx += all_dict[key]
            while cur_one_idx < len(one_idx) and one_idx[cur_one_idx] < cur_all_idx:
                if key in new_one_dict:
                    new_one_dict[key] += 1
                else:
                    new_one_dict[key] = 1
                cur_one_idx += 1
        try:
            sim_scores.append(cluster_score(new_one_dict, all_dict))
        except ZeroDivisionError:
            print(new_one_dict, all_dict)
            raise

#         sys.exit(0)
#     print(sim_scores)
#     print(one_dict, all_dict)
    return average(sim_scores), np.std(sim_scores)


def cluster_score(one_dict, all_dict):
    tp = 0
    fn = 0
    fp = 0
    total = np.sum(list(one_dict.values()))
    for key, n_total in all_dict.items():
        n_one = one_dict.get(key, 0)
        n_zero = n_total-n_one
        tp += n_one*(n_one - 1)/2
        fn += n_zero*n_one
        fp += n_one*(total-n_one)
    return tp/sqrt(1+(tp+fn)*(tp+fp))


def normalized_cluster_score(one_dict, all_dict):
    score = cluster_score(one_dict, all_dict)
    avg, sigma = simulate_score(one_dict, all_dict)
#     print(score)
#     print(score, avg, sigma)
    return (score-avg)/sigma


def get_one_all_dict(model, X, one_idx):
    if not isinstance(X, np.ndarray):
        all_prediction = model.fit_predict(X.toarray())
    else:
        all_prediction = model.fit_predict(X)
    unique, counts = np.unique(all_prediction, return_counts=True)
    all_dict = {unique[i]: counts[i] for i in range(len(unique))}
    all_counts = [all_dict.get(i, 0) for i in range(len(unique))]

    prediction = all_prediction[one_idx, ]
    unique, counts = np.unique(prediction, return_counts=True)
    one_dict = {unique[i]: counts[i] for i in range(len(unique))}
    one_counts = [one_dict.get(i, 0) for i in range(len(all_counts))]
    return one_dict, all_dict


def simulate_clusters(dataset, n_clusters=8, n_try=100):
    as_data = ASReviewData.from_file(dataset)
    _, text, labels = as_data.get_data()
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer())])
    X = text_clf.fit_transform(text)
    y = labels
#     X = X[:1000, ]
#     y = y[:1000]

    one_idx = np.where(y == 1)[0]

    with open("kmean_clustering.txt", "w") as fp:
        for _ in range(n_try):
            model = KMeans(n_clusters=n_clusters, n_init=1, n_jobs=-1)
            one_dict, all_dict = get_one_all_dict(model, X, one_idx)
            score = normalized_cluster_score(one_dict, all_dict)
            fp.write(f"{model.inertia_} {score}\n")
            fp.flush()
#             print(score, model.inertia_)


def main(dataset):
    as_data = ASReviewData.from_file(dataset)
    _, text, labels = as_data.get_data()
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer())])
    X = text_clf.fit_transform(text)
    y = labels

#     X = X[:1000, ]
#     y = y[:1000]
    one_idx = np.where(y == 1)[0]

    all_models = get_all_models()
    all_models = {"kmeans_5": all_models["kmeans_5"]}
    for model_name, model in all_models.items():
        model_fp = f"cluster_{model_name}.txt"
#         if os.path.isfile(model_fp):
#             continue
        with open(model_fp, "w") as f:
            all_prediction = model.fit_predict(X.toarray())
            unique, counts = np.unique(all_prediction, return_counts=True)
            all_dict = {unique[i]: counts[i] for i in range(len(unique))}
            all_counts = [all_dict.get(i, 0) for i in range(len(unique))]

            prediction = all_prediction[one_idx, ]
            unique, counts = np.unique(prediction, return_counts=True)
            one_dict = {unique[i]: counts[i] for i in range(len(unique))}
            one_counts = [one_dict.get(i, 0) for i in range(len(all_counts))]

            print(all_dict)
            print(one_dict)
            order = np.argsort(-np.array(one_counts))

            score = fowlkes_mallows_score(y, all_prediction)
            clust_score = cluster_score(one_dict, all_dict)
            sim_score = simulate_score(one_dict, all_dict)
            f.write(f"# {str(score)} {str(clust_score)}\n")
            print(f"{model_name} {str(score)} {str(clust_score)}")
            for i in order:
                f.write(f"{str(one_counts[i])} {str(all_counts[i])}\n")
            f.flush()


if __name__ == "__main__":
    simulate_clusters(sys.argv[1])
