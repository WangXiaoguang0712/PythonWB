#coding:utf-8
__author__ = 'T'

from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
"""
数据集名称：乳腺癌
适用模型：分类（逻辑回归）
描述：
    569个样本：{'0':212，'1':357}
    30个特征：太多不写了,可用PCA降维
    2个分类：{'0':malignant，'1':'benign'}
备注：二分类问题，分析30个特征与癌症分类的关系
"""

class Cancer(object):
    def __init__(self):
        ds = load_breast_cancer()
        self.data = ds['data']
        self.target = ds['target']
        self.feature_names = ds['feature_names']
        self.target_names = ds['target_names']#['malignant' 'benign']
        self.n_samples,self.n_features = self.data.shape #569，30
        #print ds.DESCR
        print self.data[self.target==1].shape

    #最小值为0 的特征的直方图
    def showdata_minval_0(self):
        df_cancel = pd.DataFrame(self.data,columns=self.feature_names)
        temp = df_cancel.min(axis=0)
        print temp[temp==0]

        X = self.data.T[self.feature_names == 'mean concavity'].T
        plt.figure()
        plt.hist(X,bins=20,color='r')
        plt.show()
    #能量图
    def pca_energy(self):
        from sklearn.decomposition import PCA
        nc = 6
        pca = PCA(n_components=nc)
        data_pca = pca.fit_transform(self.data)
        n_comp = pca.explained_variance_ratio_
        cum_n_comp = np.cumsum(n_comp)
        plt.plot([x+1 for x in range(nc)],cum_n_comp)
        plt.legend('upper right')
        plt.show()

    #两个特征的散点图
    def pca_scatter(self):
        print(self.target)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.data)
        plt.scatter(data_pca[:,0],data_pca[:,1],c=self.target,edgecolors='none',alpha=0.5,cmap=plt.cm.get_cmap('rainbow',2))
        plt.legend('upper right')
        plt.show()

cncr = Cancer()
#cncr.showdata_minval_0()
#cncr.pca_energy()
