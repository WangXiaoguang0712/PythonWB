# _*_ coding:utf-8 _*_
__author__ = 'T'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from twi.com.utility import plot_decision_regions
import lda


class LDA(object):
    def __init__(self, n_component = 1):
        self.n_component = n_component
        self.X_std = None
        self.eig_vals = None
        self.n_all_components = 0
        self.w = None

    def fit(self, X, y):
        sc = StandardScaler()
        self.X_std = sc.fit_transform(X)
        np.set_printoptions(precision=4)
        mean_vecs = []
        d = self.X_std.shape[1]
        self.n_all_components = d
        # 类内散步矩阵
        for label in range(1, 4):
            mean_vecs.append(np.mean(self.X_std[y == label], axis=0))
        s_w = np.zeros((d, d))
        for lable, mv in zip(range(1, 4), mean_vecs):
            class_scatter = np.cov(self.X_std[y == label].T)
            s_w += class_scatter

        # 全局散步矩阵
        s_b = np.zeros((d, d))
        mean_overall = np.mean(self.X_std, axis=0)
        for i, mean_vec in enumerate(mean_vecs):
            n = X[y == i + 1,:].shape[0]
            mean_vec = mean_vec.reshape(d, 1)
            mean_overall = mean_overall.reshape(d, 1)
            s_b += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
        # 求解特征值特征向量
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
        self.eig_vals = eig_vals
        sorted_indices = np.argsort(eig_vals)[::-1]
        self.w = eig_vecs[:,sorted_indices[:self.n_component]]

    def transform(self):
        return self.X_std.dot(self.w)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return  self.transform()

    def show_discriminablity(self):
        tot = sum(self.eig_vals)
        discr = [x / tot for x in sorted(self.eig_vals.real, reverse=True)]
        cum_discr = np.cumsum(discr)
        plt.bar(range(1, self.n_all_components + 1), discr, alpha=0.5, align='center', label='individual discr')
        plt.step(range(1, self.n_all_components + 1), cum_discr, alpha=0.9, where='mid', label='cumulative discr')
        plt.ylabel('discriminablity')
        plt.xlabel('linear discr')
        plt.ylim([-0.1, 1.1])
        plt.legend(loc='best')
        plt.show()

if __name__ == "__main__":
    df_wine = pd.read_csv('data/wine.csv', header=None)
    X, y = df_wine.values[:, 1:], df_wine.values[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    lda = LDA(n_component=2)
    lda.fit(x_train, y_train)
    x_train_pca = lda.transform()
    """
    model = lda.LDA(n_topics=2)
    x_train_pca = model.fit_transform(X, y)
    """
    lr = LogisticRegression()
    lr.fit(x_train_pca, y_train)
    plot_decision_regions(x_train_pca.real, y_train, classfier=lr)
    plt.show()