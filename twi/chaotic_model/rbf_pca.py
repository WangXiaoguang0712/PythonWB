# _*_ coding:utf-8 _*_
__author__ = 'T'

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.linalg import eigh
import numpy as np

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def rbf_kernal_pca(X, gamma, n_components):
    '''
    :param X:  shape = [n_sample, n_feature]
    :param gamma: Tuning parameter
    :param n_components: number of component to return
    :return:X_pc shape= [n_sample, k_feature]
    '''
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = np.exp(- gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    print(eigvals)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]
    return X_pc, lambdas

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - x) ** 2) for x in X])
    k = np.exp(-gamma * pair_dist)
    return  k.dot(alphas / lambdas)

if __name__ == "__main__":
    X, y = make_moons(n_samples=100, random_state=123)
    skpca = PCA(n_components=2)
    X_spca = skpca.fit_transform(X)
    X_pca, lambdas = rbf_kernal_pca(X, gamma=15, n_components=2)

    x_new = X[25]
    x_proj = X_pca[25]
    x_reproj = project_x(x_new, X, gamma=15, alphas=X_pca, lambdas=lambdas)
    print(x_reproj)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='r', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='b', marker='o', alpha=0.5)
    ax[1].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='r', marker='^', alpha=0.5)
    ax[1].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='b', marker='o', alpha=0.5)
    plt.show()
