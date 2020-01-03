#!/usr/bin/env python

from math import log
import os
import sys

from asreview.models.embedding import load_embedding
from asreview.readers import ASReviewData
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from sklearn.cluster import KMeans
from cluster import get_one_all_dict, normalized_cluster_score
from os.path import splitext
from hdbscan import HDBSCAN
from sklearn.mixture.gaussian_mixture import GaussianMixture
# 
# def read_embedding(embedding_fp):
#     embedding = {}
#     with open(embedding_fp, 'r') as f:
#         f.readline()
#         for line in f:
#             line = line.rstrip()
#             values = line.split('')
#             

def get_freq_dict(all_text):
    text_dicts = []
    for text in all_text:
        cur_dict = {}
        word_sequence = text_to_word_sequence(text)
        for word in word_sequence:
            if word in cur_dict:
                cur_dict[word] += 1
            else:
                cur_dict[word] = 1
        text_dicts.append(cur_dict)
    return text_dicts


def get_idf(text_dicts):
    all_count = {}
    for text in text_dicts:
        for word in text:
            if word in all_count:
                all_count[word] += 1
            else:
                all_count[word] = 1

    idf = {}
    for word in all_count:
        idf[word] = log(len(text_dicts)/all_count[word])
    return idf


def get_X_from_dict(text_dicts, idf, embedding):
    n_vec = len(embedding[list(embedding.keys())[0]])
    X = np.zeros((len(text_dicts), n_vec))
    for i, text in enumerate(text_dicts):
        text_vec = None
        for word in text:
            cur_count = text[word]
            cur_idf = idf[word]
            cur_vec = embedding.get(word, None)
            if cur_vec is None:
                continue
            if text_vec is None:
                text_vec = cur_vec*cur_idf*cur_count
            else:
                text_vec += cur_vec*cur_idf*cur_count
        if text_vec is None:
            text_vec = np.random.random(n_vec)
        text_vec /= np.linalg.norm(text_vec)
        X[i] = text_vec
    return X


def get_Xy(data_fp, embedding_fp):
    embedding = load_embedding(embedding_fp, n_jobs=-1)
    as_data = ASReviewData.from_file(data_fp)
    _, text, labels = as_data.get_data()
    text_counts = get_freq_dict(text)
    idf = get_idf(text_counts)
    X = get_X_from_dict(text_counts, idf, embedding)
    y = labels
    return X, y


def simulate_clusters(data_fp, embedding_fp, n_clusters=8, n_try=50):
    X, y = get_Xy(data_fp, embedding_fp)
#     X = X[:1000, ]
#     y = y[:1000]

    data_name = splitext(os.path.basename(data_fp))[0]
    one_idx = np.where(y == 1)[0]

    filename = os.path.join("cluster_data", data_name, f"embed_k{n_clusters}")
    os.makedirs(os.path.join("cluster_data", data_name), exist_ok=True)
    if os.path.isfile(filename):
        return
    with open(filename, "w") as fp:
        for _ in range(n_try):
            model = KMeans(n_clusters=n_clusters, n_init=1, n_jobs=-1)
            one_dict, all_dict = get_one_all_dict(model, X, one_idx)
            score = normalized_cluster_score(one_dict, all_dict)
            fp.write(f"{model.inertia_} {score}\n")
            fp.flush()
#             print(score, model.inertia_)


def simulate_hdbscan(data_fp, embedding_fp):
    X, y = get_Xy(data_fp, embedding_fp)
#     X = X[:1000, ]
#     y = y[:1000]

#     data_name = splitext(os.path.basename(data_fp))[0]
    one_idx = np.where(y == 1)[0]

#     filename = os.path.join("cluster_data", data_name, f"embed_k{n_clusters}")
#     os.makedirs(os.path.join("cluster_data", data_name), exist_ok=True)
#     if os.path.isfile(filename):
#         return
#     with open(filename, "w") as fp:
    for mcs in [2, 3, 5, 7, 10]:
        for ms in [1, 3, 5, 7, 10]:
            model = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            one_dict, all_dict = get_one_all_dict(model, X, one_idx)
            score = normalized_cluster_score(one_dict, all_dict)
            print(mcs, ms, score, one_dict, all_dict)
#             fp.write(f"{model.inertia_} {score}\n")
#             fp.flush()
#             print(score, model.inertia_)


def simulate_gaussian(data_fp, embedding_fp, n_clusters=8, n_try=10):
    X, y = get_Xy(data_fp, embedding_fp)
#     X = X[:1000, ]
#     y = y[:1000]

    data_name = splitext(os.path.basename(data_fp))[0] + "_gauss"
    one_idx = np.where(y == 1)[0]
    filename = os.path.join("cluster_data", data_name, f"embed_k{n_clusters}")
    os.makedirs(os.path.join("cluster_data", data_name), exist_ok=True)
    if os.path.isfile(filename):
        return

    with open(filename, "w") as fp:
        for _ in range(n_try):
            model = GaussianMixture(n_components=n_clusters)
            one_dict, all_dict = get_one_all_dict(model, X, one_idx)
            score = normalized_cluster_score(one_dict, all_dict)
            fp.write(f"{model.lower_bound_} {score}\n")
            fp.flush()


def main(data_fp, embedding_fp):
    embedding = load_embedding(embedding_fp, n_jobs=-1)
    as_data = ASReviewData.from_file(data_fp)
    _, text, labels = as_data.get_data()
    text_counts = get_freq_dict(text)
    idf = get_idf(text_counts)
    X = get_X_from_dict(text_counts, idf, embedding)
    print(X)
#     print(len(text))


if __name__ == '__main__':
#     main(sys.argv[1], sys.argv[2])
#     for n_clusters in [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 25, 30]:
#         simulate_clusters(sys.argv[1], sys.argv[2], n_clusters=n_clusters)
#     simulate_hdbscan(sys.argv[1], sys.argv[2])
    for n_clusters in [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 25, 30]:
        simulate_gaussian(sys.argv[1], sys.argv[2], n_clusters=n_clusters)