import numpy as np
from scipy.spatial.distance import cdist

def prod_non_zero_diag(x):
    return np.prod(np.diag(x)[np.diag(x) != 0])

def are_multisets_equal(x, y):
    return np.array_equal(np.sort(x), np.sort(y))


def max_after_zero(x):
    return np.max(x[1:][np.nonzero(x[:-1] == 0)])


def convert_image(img, coefs):
    return np.dot(img, coefs).astype(np.uint8)


def run_length_encoding(x):
    if x.size == 0:
        return np.array([]), np.array([])
    a = np.concatenate(([0], np.where(x[:-1] != x[1:])[0] + 1))
    b = np.diff(np.concatenate((a, [x.size])))
    return x[a], b


def pairwise_distance(x, y):
    return np.sqrt(np.sum((x[:, np.newaxis] - y) ** 2, axis=-1))

def pairwise_distance_scipy(x, y):
    return cdist(x, y, metric='euclidean')