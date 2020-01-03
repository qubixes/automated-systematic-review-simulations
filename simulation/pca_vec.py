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
from gensim.corpora.dictionary import Dictionary
from asreview.models.embedding import load_embedding
from math import log
from sklearn.decomposition import PCA
from tqdm._tqdm import tqdm


def build_filename(data_fp, n_clusters, method_name="d2v"):
    data_name = splitext(os.path.basename(data_fp))[0] + "_" + method_name
    file_out = os.path.join("cluster_data", data_name, f"embed_k{n_clusters}")
    os.makedirs(os.path.join("cluster_data", data_name), exist_ok=True)
    return file_out


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def create_pca_X(texts, embedding, pca_str=0):

#     texts = texts[:10]
#     corpus = [TaggedDocument(simple_preprocess(text), [i])
#               for i, text in enumerate(texts)]

    plain_corpus = [simple_preprocess(text)
                    for i, text in enumerate(texts)]

    corpus_dict = Dictionary(documents=plain_corpus)
    dfs = corpus_dict.dfs
    ids = corpus_dict.token2id
    idf = {}
    for word, token_id in ids.items():
        idf[word] = log(len(plain_corpus)/dfs[token_id])


    X_doc = []
    explained_var_doc = []
    word_mean_doc = []
    pca_component_doc = []
    embedding_vec_size = len(embedding[list(embedding)[0]])
    for doc in tqdm(plain_corpus):
        X = []
        avg_word = np.zeros((embedding_vec_size,))
        for word in doc:
            new_vec = embedding[word]*idf[word]
            avg_word += new_vec
            X.append(new_vec)

        if len(doc) <= 1:
            X_doc.append(np.zeros((1, embedding_vec_size)))
            explained_var_doc.append(0)
            pca_component_doc.append(np.zeros((embedding_vec_size,)))
            if len(doc) == 1:
                word_mean_doc.append(new_vec)
            else:
                word_mean_doc.append(np.zeros((embedding_vec_size,)))
            continue

#         if len(doc) == 1:
#             X_doc.append()

        avg_word /= len(doc)
        X = np.array(X)
        n_components = min(X.shape[0], 5)
        pca_model = PCA(n_components=n_components)
        pca_model.fit(X)
        components = pca_model.components_
        X_doc.append(np.multiply(
            components, pca_model.explained_variance_.reshape((-1, 1))))
        pca_component_doc.extend([x for x in components])
        word_mean_doc.append(avg_word)
        explained_var_doc.extend(pca_model.explained_variance_.tolist())

    print("Create arrays")
    explained_var_doc = np.array(explained_var_doc)
    pca_component_doc = np.array(pca_component_doc)

#     print(pca_component_doc)
#     print("===================")
    print("Multiply matrix", pca_component_doc.shape, explained_var_doc.reshape((-1, 1)).shape)
    pca_component_doc = np.multiply(
        pca_component_doc, explained_var_doc.reshape((-1, 1)))
#     print(pca_component_doc)
#     print("+++++++++++++++")
#     print(explained_var_doc)
    print("Define PCA")
    global_pca_model = PCA()
    print(pca_component_doc.shape)
    global_pca_model.fit(pca_component_doc)
    base = global_pca_model.components_
    print(f"Found base.")
    print(f"explained variance: {global_pca_model.explained_variance_}")

    final_doc_X = []
    for i, X in enumerate(X_doc):
        vec = np.sum(X.dot(base.T), axis=0)
        final_vec = np.append(word_mean_doc[i], np.absolute(pca_str*vec))
        final_norm = np.linalg.norm(final_vec)
        if final_norm > 1e-7:
            final_vec /= np.linalg.norm(final_vec)
        print(final_vec)
        if i > 10:
            sys.exit()
        final_doc_X.append(final_vec)

    return np.array(final_doc_X)


def pca_clusters(data_fp="ptsd.csv", n_clusters=8, n_run=1, vector_size=64):
    _, texts, labels = asreview.ASReviewData.from_file(data_fp).get_data()
    embedding_fp = splitext(data_fp)[0] + ".vec"
    embedding = load_embedding(embedding_fp)

#     texts = texts[:1000]
#     labels = labels[:1000]

    X = create_pca_X(texts, embedding, pca_str=1e7)
    one_idx = np.where(labels == 1)[0]
    for _ in range(5):
        kmeans_model = KMeans(n_clusters=n_clusters, n_init=1)
        one_dict, all_dict = get_one_all_dict(kmeans_model, X, one_idx)
#         print(one_dict, all_dict)
        score = normalized_cluster_score(one_dict, all_dict)
        print(f"{score}")

#     filename = build_filename(data_fp, n_clusters, "d2v_" + str(vector_size))
#     with open(filename, "w") as fp:
#         for _ in range(n_run):
#             model = gensim.models.doc2vec.Doc2Vec(
#                 vector_size=vector_size, min_count=2, epochs=40)
#             model.build_vocab(corpus)
#             model.train(corpus, total_examples=model.corpus_count,
#                         epochs=model.epochs)
# 
#             X = []
#             for doc_id in range(len(corpus)):
#                 X.append(model.infer_vector(corpus[doc_id].words))
# 
#             X = np.array(X)
#             one_idx = np.where(labels == 1)[0]
#             kmeans_model = KMeans(n_clusters=n_clusters, n_init=1)
#             one_dict, all_dict = get_one_all_dict(kmeans_model, X, one_idx)
#             score = normalized_cluster_score(one_dict, all_dict)
# 
#             fp.write(f"{np.random.rand()} {score}\n")
#             fp.flush()


if __name__ == "__main__":
    pca_clusters()
