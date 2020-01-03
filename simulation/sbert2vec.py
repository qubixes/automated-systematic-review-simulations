#!/usr/bin/env python

import numpy as np
# from topic_modeling import create_corpus
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
from cluster import get_one_all_dict, normalized_cluster_score
import logging
import gensim
import asreview
import sys
from os.path import splitext
import os
from sentence_transformers.SentenceTransformer import SentenceTransformer
from tqdm import tqdm


def build_filename(data_fp, n_clusters, method_name="d2v"):
    data_name = splitext(os.path.basename(data_fp))[0] + "_" + method_name
    file_out = os.path.join("cluster_data", data_name, f"embed_k{n_clusters}")
    os.makedirs(os.path.join("cluster_data", data_name), exist_ok=True)
    return file_out


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def sbert_clusters(data_fp="ptsd.csv", n_run=10):
    _, texts, labels = asreview.ASReviewData.from_file(data_fp).get_data()

    model = SentenceTransformer('bert-base-nli-mean-tokens')
#     X = []
#     for text in tqdm(texts):
#         X.appmodel.encode(text))

    texts = texts
    labels = labels

    X = np.array(model.encode(texts))
    print(X.shape)
    print(X)

    one_idx = np.where(labels == 1)[0]
    for n_clusters in [2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 25, 30]:
        filename = build_filename(data_fp, n_clusters, "sbert")
        with open(filename, "w") as fp:
            for _ in range(n_run):
                kmeans_model = KMeans(n_clusters=n_clusters, n_init=1)
                one_dict, all_dict = get_one_all_dict(kmeans_model, X, one_idx)
                score = normalized_cluster_score(one_dict, all_dict)
                fp.write(f"{np.random.rand()} {score}\n")
                fp.flush()


if __name__ == "__main__":
    sbert_clusters(sys.argv[1])
