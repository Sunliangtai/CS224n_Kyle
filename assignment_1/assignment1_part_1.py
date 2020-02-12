# All Import Statements Defined Here
# Note: Do not add to this list.
# All the dependencies you need, can be installed by running .
# ----------------

import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

import time
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import nltk
#nltk.download('reuters')
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)
# ----------------


def read_corpus(category="crude"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    time_s = time.time()
    print("start reading")
    print("-" * 80)
    files = reuters.fileids(category)
    time_e = time.time()
    print("finish reading")
    print("time cost " + str(time_e - time_s) + "s")
    print("-" * 80)
    return [[START_TOKEN] + [w.lower()
                             for w in list(reuters.words(f))] + [END_TOKEN]
            for f in files]


def distinct_words(corpus):

    time_s = time.time()
    print("start distincting words")
    print("-" * 80)

    corpus_words = []
    num_corpus_words = -1

    for corpu in corpus:
        for content in corpu:
            corpus_words.append(content)
    corpus_words = sorted(list(set(corpus_words)))
    num_corpus_words = len(corpus_words)
    time_e = time.time()
    print("finish distincting words")
    print("time cost " + str(time_e - time_s) + "s")
    print("-" * 80)
    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    words, num_words = distinct_words(corpus)
    time_s = time.time()
    print("start computing co-occurrence matrix")
    print("-" * 80)
    M = np.zeros((num_words, num_words))
    word2Ind = {}

    word2Ind = {words[idx]: idx for idx in range(num_words)}
    for corpu in corpus:
        for idx in range(len(corpu)):
            for i in range(max(0, idx - window_size),
                           min(idx + window_size + 1, len(corpu))):
                if idx != i:
                    M[word2Ind[corpu[idx]]][word2Ind[corpu[i]]] += 1
    time_e = time.time()
    print("finish computing co-occurrence matrix")
    print("time cost " + str(time_e - time_s) + "s")
    print("-" * 80)
    return M, word2Ind


def reduce_to_k_dim(M, k=2):
    time_s = time.time()
    n_iters = 10
    M_reduced = None

    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    print("-" * 80)
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = svd.fit(M).transform(M)

    time_e = time.time()
    print("Done.")
    print("time cost " + str(time_e - time_s) + "s")
    print("-" * 80)
    return M_reduced


def plot_embeddings(M_reduced, word2Ind, words):
    time_s = time.time()
    print("start plotting")
    print("-" * 80)

    for word in words:
        idx = word2Ind[word]
        x = M_reduced[idx][0]
        y = M_reduced[idx][1]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x + 0.0001, y, word, fontsize=9)
    time_e = time.time()
    print("finish")
    print("time cost " + str(time_e - time_s) + "s")
    print("-" * 80)

    plt.show()


# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------
time_start = time.time()
reuters_corpus = read_corpus()
M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
    reuters_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:,
                                                   np.newaxis]  # broadcasting

words = [
    'barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil',
    'output', 'petroleum', 'venezuela'
]
plot_embeddings(M_normalized, word2Ind_co_occurrence, words)
time_end = time.time()
print("all time cost " + str(time_end - time_start) + "s")
