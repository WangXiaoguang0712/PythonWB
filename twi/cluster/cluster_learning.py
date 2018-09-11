# _*_ coding:utf-8 _*_
__author__ = 'T'

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def make_data(is_show=True):
    X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
    if is_show:
        plt.scatter(X[:, 0], X[:, 1], c='black', marker='o', s=50)
        plt.show()
    return X, y

def test_kmeans():
    X, y = make_data(False)
    km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    # n_clusters: 中心点；n_init： 随机生成10组中心点计算10次，选择最好MSE最小的一次；max_iter: 最大循环次数
    # tol:容忍度，判定是是是否收敛，防止无法收敛的情况。
    # init: 初始化中心点的方法，取值可为 kmean++
    y_km = km.fit_predict(X)
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', marker='s', label='cluster 1')
    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', marker='o', label='cluster 2')
    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue', marker='v', label='cluster 3')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='*', s=250, label='centroids')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    test_kmeans()