# _*_ coding:utf-8 _*_
__author__ = 'T'

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

def make_data(is_show=True):
    X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
    if is_show:
        plt.scatter(X[:, 0], X[:, 1], c='black', marker='o', s=50)
        plt.show()
    return X, y


def test_kmeans():
    '''
    KMean 示例
    :return:
    '''
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


def test_elbow():
    '''
    确定合适的簇数。
    :return:
    '''
    X, y = make_data(False)
    dist = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(X, y)
        dist.append(km.inertia_)
    plt.plot([ x for x in range(1, 11)], dist, c='lightblue', marker='o', label='elbow')
    plt.xlabel('number of cluesters')
    plt.ylabel('dist')
    plt.show()


def test_agglomerative():
    '''
    层次聚类
    :return:
    '''
    np.random.seed(123)
    variables = ['X', 'Y', 'Z']
    labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
    X = np.random.random_sample([5, 3]) * 10
    df = pd.DataFrame(X, columns=variables, index=labels)
    dist = pdist(X, metric='euclidean')  # pdist: 两两行向量之间的距离，返回1 * n 的向量
    sf = squareform(dist)  # squareform 将pdist 的结果转变为对称矩阵的形式。
    df_row_dist = pd.DataFrame(sf, columns=labels, index=labels)
    row_cluster = linkage(df_row_dist.values, method='complete', metric='euclidean')
    df_row_cluster = pd.DataFrame(row_cluster,
                                  columns=['row label 1', 'row_label 2', 'distance', 'no in cluster'],
                                  index=[ 'cluster %s' % (i + 1) for i in range(row_cluster.shape[0])])
    # print(df_row_cluster)  # 每一行代表一次合并
    """
    row_dendr = dendrogram(row_cluster, labels=labels)
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    plt.show()
    """
    fig = plt.figure(figsize=(6, 6))
    axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    row_dendr = dendrogram(row_cluster, orientation='left')
    df_rowclust = df.ix[row_dendr['leaves'][::-1]]
    print(df_rowclust)
    axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
    axd.set_xticks([])
    axd.set_yticks([])

    for i in axd.spines.values():
        i.set_visible(False)  # 边框
    fig.colorbar(cax)
    axd.set_title('axd')
    axm.set_title('axm')
    axm.set_xticklabels([''] + list(df_rowclust.columns))
    axm.set_yticklabels([''] + list(df_rowclust.index))
    print([''] + list(df_rowclust.columns))
    plt.show()


def test_dbscan():
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    km = KMeans(n_clusters=2, random_state=0)
    y_km = km.fit_predict(X)
    ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c='lightblue', marker='o')
    ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c='lightgreen', marker='v')
    ax1.set_title('kmean clustering')
    ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
    y_ac = ac.fit_predict(X)
    ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue', marker='o')
    ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='lightgreen', marker='v')
    ax2.set_title('Agglomeraive clustering')
    db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    y_db = db.fit_predict(X)
    ax3.scatter(X[y_db == 0, 0], X[y_db == 0, 1], c='lightblue', marker='o')
    ax3.scatter(X[y_db == 1, 0], X[y_db == 1, 1], c='lightgreen', marker='v')
    ax3.set_title('DBSCAN clustering')

    plt.show()



if __name__ == "__main__":
    test_dbscan()