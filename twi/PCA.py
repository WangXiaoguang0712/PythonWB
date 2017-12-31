#coding:utf-8
__author__ = 'T'

import numpy  as np
import matplotlib.pyplot as plt

class PCA(object):
    def __init__(self,n_component=0,n_all_component=0,components_ = [],contribution_ = [],accurency = 0.95):
        self.n_component = n_component
        self.n_all_component = n_all_component
        self.components_ = components_
        self.contribution_ = contribution_
        self.accurency = accurency

    def zeromean(self,mat):
        mean_val = np.mean(mat,axis=0)
        new_data = mat - mean_val
        return new_data,mean_val

    def percent2n(self,eig_val_arr,pct):
        sorted_arr = np.sort(eig_val_arr)[::-1]
        sumarray = sum(sorted_arr)
        cum_sum_arr = np.cumsum(sorted_arr)
        self.contribution_ = cum_sum_arr*1.0/sumarray
        for idx,item in enumerate( self.contribution_):
            if item>=pct:
                self.n_component = idx+1
                return

    def showcontribution(self):
        print self.contribution_
        plt.figure()
        plt.subplot(111)
        X = [x for x in range(self.n_all_component+1)]
        Y = np.concatenate(([0],self.contribution_))
        plt.plot(X,Y,color='r')
        plt.annotate('best num of component',xy=(X[self.n_component],Y[self.n_component]),xytext=(X[self.n_component],0.7),arrowprops=dict(facecolor='black',shrink=0.5))
        plt.xlabel('conponents')
        plt.ylabel('contribution')
        plt.xlim(0,self.n_all_component)
        plt.ylim([0,1])
        plt.grid(True)
        plt.show()

    def fit(self,mat):
        self.n_all_component = mat.shape[1]
        new_data,mean_val = self.zeromean(mat)
        cov_mat = np.cov(new_data,rowvar=0) #rowvar=0 表示每行为一个样本
        eig_val,eig_vec = np.linalg.eig(cov_mat) #求特征值和特征向量
        sort_indice = np.argsort(-eig_val) #返回数组中数值从大道小的数据的索引
        self.components_ = eig_vec[sort_indice] #所有特征向量
        self.percent2n(eig_val,self.accurency)

        n_eig_val = sort_indice[0:self.n_component] #前n个特征向量
        n_eig_vec = eig_vec[n_eig_val] #前n维向量
        low_d_mat = np.dot(new_data,n_eig_vec.T) #低维度数据
        rec_mat = np.dot(low_d_mat,n_eig_vec)+mean_val #重构数据
        return low_d_mat,rec_mat


pca = PCA(accurency=0.8)
data = np.random.random((5,3))
val = pca.fit(data)
pca.showcontribution()

