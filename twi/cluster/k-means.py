#coding:utf-8
__author__ = 'T'

import copy
import  numpy as np
from  matplotlib import pyplot as plt

class KMean(object):
    def __init__(self):
        self.convergence_step = 0
        self.ori_center = np.zeros((k,2))

    def initcenter(self,ds,k):
        """
        :param ds: 待聚类数据集
        :param k: 聚类数目
        :return:聚类中心
        """
        n_samples,n_features = ds.shape
        center_point = np.zeros((k,2))
        for i in range(k):
            rnum = np.random.randint(0,n_samples)
            center_point[i,:] = ds[rnum,:]
        return center_point

    def calcdist(self,v1,v2):
        """
        :param v1: 向量1
        :param v2: 向量1
        :return:距离
        """
        return np.sqrt(sum(np.power(v1 - v2,2)))

    def analyze(self,ds,k):
        """
        :param ds: 待聚类数据集
        :param k: 聚类数目
        :return:none
        """
        n_samples,n_features = ds.shape
        resarray = np.zeros((n_samples,2))
        center_point = self.initcenter(ds,k)
        self.ori_center = copy.deepcopy(center_point[:,:])
        centerchanged = True
        while centerchanged:
            self.convergence_step += 1
            centerchanged = False
            for i in range(n_samples):
                for j in range(k):
                    #计算每个点到中心点的距离
                    dist = self.calcdist(ds[i,:],center_point[j,:])
                    if j == 0:
                        cluser_id,mindist = j,dist
                    else:
                        if dist < mindist:
                            cluser_id,mindist = j,dist
                #如果中心点发生变化，则说明未收敛，需要继续迭代
                if resarray[i,0] != cluser_id:
                    centerchanged = True
                    resarray[i,:] = (cluser_id,mindist)
            #所有点都完成一次遍历后，更新中心点
            for j in range(k):
                points_in_cluster = ds[resarray[:,0]==j]
                center_point[j,:] = np.mean(points_in_cluster,axis=0)
        print 'analyze over!'
        return center_point,resarray

    def showcluster(self,ds,k,center_point,resarray):
        """
        :param ds: 数据集
        :param k: 聚类数
        :param center_point: 中心点
        :param resarray: 分类及距离
        :return:None
        """
        n_sample = ds.shape[0]
        mark = ['.r', '.g', '.b', '.k']

        if k > len(mark):
            print 'sorry,k is too big for me!'
            return 1
        print self.ori_center
        print center_point
        plt.figure()
        #画所有点
        for i in xrange(n_sample):
            cluster_index =int(resarray[i,0])
            plt.plot(ds[i,0],ds[i,1],mark[cluster_index])

        #画中心点
        mark1 = ['r','g','b','k']
        for j in range(k):
            plt.scatter(center_point[j,0],center_point[j,1],marker = '+',color=mark1[j])
            plt.scatter(self.ori_center[j,0],self.ori_center[j,1],marker = 'D',color=mark1[j])
        plt.show()

if __name__ == "__main__" :
    print 'stpe 1: load data'
    ds = []
    k = 4
    file = 'sample.txt'
    fr = open(file)
    for line in fr.readlines():
        linearr = line.strip().split(',')
        ds.append([float(linearr[0]),float(linearr[1])])
    ds = np.array(ds)
    print 'step 2 : anlyze'
    mysample = KMean()
    center_point,resarray = mysample.analyze(ds,k)
    #print center_point,resarray
    print 'sept 3: show graph'
    print mysample.convergence_step
    mysample.showcluster(ds,k,center_point,resarray)
