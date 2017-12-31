#coding:utf-8
__author__ = 'T'

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import  load_iris #鸢尾花
from twi.regression.dm_regression_linear import twi_linear_regression
"""
数据集名称：鸢尾花
适用模型：分类
描述：
    150个样本：['0'：50,'1'：50,'2'：50]
    4个特征：['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']；sepal（花萼）；petal（花瓣）
    3个分类：['0'：'setosa','1'：'versicolor','2'：'virginica']
备注：3种鸢尾花的花萼长宽以及花瓣长宽的数据
"""
class Iris(object):
    def __init__(self):
        ds = load_iris()
        self.data = ds['data']
        self.target = ds['target']
        self.features = ds['feature_names']#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        self.target_names = ds['target_names']#['setosa' 'versicolor' 'virginica']
        self.n_samples,self.n_features = self.data.shape#(150L, 4L)
        #se_iris = pd.Series(np.concatenate((self.features,['target']),axis=0))
        #df_iris = pd.DataFrame(np.concatenate((self.data,self.target.reshape(-1,1)),axis=1),columns=se_iris)

    def showdata_scatter(self):
        plt.figure()
        plt.subplot(121)
        color_ = ['r','g','b']
        marker_ = ['.','v','+']
        xindex = 0
        for l,m,c in zip(range(len(self.target_names)),marker_,color_):
            plt.scatter(self.data[self.target==l,xindex],self.data[self.target==l,xindex+1],c=c,marker=m,label=self.target_names[l])
        plt.xlabel(self.features[xindex])
        plt.ylabel(self.features[xindex+1])
        plt.legend(loc='upper right')

        plt.subplot(122)
        for label in range(len(self.target_names)):
            plt.scatter(self.data[self.target==label,xindex+2],self.data[self.target==label,xindex+3],c=color_[label],marker=marker_[label],label=self.target_names[label])
        plt.xlabel(self.features[xindex+2]);plt.ylabel(self.features[xindex+3])
        plt.legend(loc='lower right')
        plt.show()

    def showdata_hist(self):
        plt.figure()
        plt.subplot(111)
        color = ['r','g','b']
        xindex = 3
        for label,color_ in zip(range(len(self.target_names)),color):
            plt.hist(self.data[self.target==label,xindex],label=self.target_names[label],color=color_)
        plt.legend(loc='upper right')
        plt.xlabel(self.features[xindex])
        plt.show()

    def test(self):
        x = np.linspace(-np.pi,np.pi,256)
        c,s = np.cos(x),np.sin(x)
        #plt.scatter(x,c,c='r',marker='.')
        #plt.scatter(x,s,c='g',marker='.')

        plt.plot(x,c,c='r',linestyle='-',label='cos')
        plt.plot(x,s,c='g',linestyle='-',label='sin')
        plt.legend(loc='upper left')
        plt.xlabel('ss')
        plt.show()

#显示数据
#iris = Iris()
#iris.showdata_scatter()

"""
#预测
lr = twi_linear_regression()
iris_data,iris_feature = lr.getDataSet()

# 梯度下降算法
lr.fit(iris_data,'GD')
print lr.theta
plt.figure()
plt.plot(lr.J)
plt.ylabel('lost')
plt.xlabel('iter count')
plt.title('convergence graph')
plt.show()

print lr.predict(iris_data[148:150,0:3])
print iris_data[148:150,3]

#方程式算法

lr.fit(iris_data,1)
print lr.theta
print lr.predict(iris_data[148:150,0:3])
print iris_data[148:150,3]
"""
