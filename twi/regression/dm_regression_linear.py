#coding:utf-8
__author__ = 'T'
import  numpy as np
import  pandas as pd
import math
import random
import  matplotlib.pyplot as plt
from sklearn import datasets

class twi_linear_regression(object):
    theta = None
    J  = 0
    def getDataSet(self):
        dataset  = datasets.load_iris()
        iris_feature = dataset['feature_names']
        iris_data = dataset['data']
        return iris_data,iris_feature

    #方程式法
    def getThetaByEquation(self,dataset):
        theta = 0
        ds_rows,ds_cols = dataset.shape

        dataset_ext = np.ones([ds_rows,1],dtype=int)# 创建全1矩阵
        X = np.concatenate((dataset_ext,dataset[:,0:ds_cols-1]),axis=1)

        y = dataset[:,-1]
        theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)# theta = (X'X)^(-1)X'Y
        return theta

    #梯度下降法优化方式之归一化
    def feature_normalize(self,X):
        rows,cols = X.shape
        mu = np.zeros([1,cols])
        sigma = np.zeros([1,cols])
        for i in range(cols):
            mu[0,i] = np.mean(X[:,i])
            sigma[0,i] = np.std(X[:,i])
        X = (X-mu)/sigma
        return X

    #梯度下降法
    def getThetaByGradientDecent(self,dataset):
        theta = 0
        dataset = self.feature_normalize(dataset)
        ds_rows,ds_cols = dataset.shape
        dataset_ext = np.ones([ds_rows,1])# 创建全1矩阵

        #初始化 及步长
        theta = np.zeros([ds_cols,1])
        alpha = 1
        X = np.concatenate((dataset_ext,dataset[:,0:ds_cols-1]),axis=1)
        y = dataset[:,-1].reshape(ds_rows,1)
        J = pd.Series(np.arange(100,dtype=float))# 1-800的序列
        for i in range(1000):
            #theta = theta - (alpha/m) * (X.T.dot(X.dot(theta) - y))
            alpha = alpha/(i+1) + 0.001
            theta = theta - alpha/ds_rows*(X.T.dot(X.dot(theta)-y))
            J[i] = 0.5*np.sum((X.dot(theta)-y)**2)/(2*ds_rows) #计算损失函数值
        return  theta,J
    #分裂数据集（其实没必要）
    def splitDataSet(self,dataset,ratio):
        ds_r,ds_c = dataset.shape
        copy = dataset[:,:]

        training_set = np.array([])
        for i in range(int(len(dataset)*ratio)):
            j = random.randrange(len(copy))
            item = copy[j,:].reshape(1,ds_c)
            if i == 0:
                training_set = item
            else:
                training_set = np.concatenate((training_set,item),axis=0)
            copy = np.delete(copy,j,0)
        return [training_set,copy]
    #拟合求参数
    def fit(self,dataset,method):
        if method == 'GD':
            self.theta,self.J = self.getThetaByGradientDecent(dataset)
        else:
            self.theta = self.getThetaByEquation(dataset)
    #预测
    def predict(self,vector):
        rows,cols = vector.shape
        vector_ext = np.ones([rows,1])
        X = np.concatenate((vector_ext,vector),axis=1)
        return  np.dot(X,self.theta)


