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
import json


def doc2vec(data_fp, doc_vec_fp, vector_size=64):
    _, texts, _ = asreview.ASReviewData.from_file(data_fp).get_data()

    corpus = [TaggedDocument(simple_preprocess(text), [i])
              for i, text in enumerate(texts)]

    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=vector_size, min_count=2, epochs=40)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count,
                epochs=model.epochs)

    X = []
    for doc_id in range(len(corpus)):
        doc_vec = model.infer_vector(corpus[doc_id].words)
        X.append(doc_vec.tolist())

    with open(doc_vec_fp, "w") as fp:
        json.dump(X, fp)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        vector_size = int(sys.argv[3])
    else:
        vector_size = 64
    doc2vec(sys.argv[1], sys.argv[2], vector_size=vector_size)
