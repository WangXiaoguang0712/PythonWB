#coding:utf-8
__author__ = 'T'

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from twi.com.utility import  plot_decision_regions

class PCA(object):
    def __init__(self,n_component=1):
        self.n_component = n_component
        self.contribution_ = None
        self.n_all_component = 0


    def showcontribution(self, X):
        print(self.contribution_)
        plt.figure()
        plt.subplot(111)
        xxx = range(1, self.n_all_component + 1)

        tot = sum(self.contribution_)
        val_sorted = [x / tot for x in sorted(self.contribution_, reverse=True)]
        val_cum = np.cumsum(val_sorted)
        print(val_cum)
        plt.bar(xxx, val_sorted, color='r', align='center', label='individual explained variance')
        plt.step(xxx, val_cum, color='blue', where='mid', label='cumulative explained variance')
        plt.xlabel('principal components')
        plt.ylabel('contribution')
        plt.xlim(0,self.n_all_component + 1)
        plt.ylim([0,1])
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

    def fit(self, X):
        self.n_all_component = X.shape[1]
        sc = StandardScaler()
        X_std = sc.fit_transform(X)
        X_cov = np.cov(X_std,rowvar=0)  # rowvar=0 表示每行为一个样本
        eig_val, eig_vec = np.linalg.eig(X_cov)  # 求特征值和特征向量
        sort_indice = np.argsort(-eig_val)  # 返回数组中数值从大道小的数据的索引
        self.contribution_ = eig_val  # 所有特征向量

        n_eig_val = sort_indice[0 : self.n_component]  # 前n个特征向量
        self.dest_eig_vec = eig_vec[:, n_eig_val]  # 前n维向量


    def transform(self, X):
        sc = StandardScaler()
        X_std = sc.fit_transform(X)
        X_pca = np.dot(X_std, self.dest_eig_vec)  # 降维数据
        return  X_pca

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    df_wine = pd.read_csv('data/wine.csv', header=None)
    X, y = df_wine.values[:, 1:], df_wine.values[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    pca = PCA(n_component=2)
    pca.fit(X)
    pca.showcontribution(x_train)
    x_train_pca = pca.transform(x_train)
    lr = LogisticRegression()
    lr.fit(x_train_pca, y_train)
    plot_decision_regions(x_train_pca, y_train, classfier=lr)
    plt.show()

