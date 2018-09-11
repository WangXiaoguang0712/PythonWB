#coding:utf-8
import copy
import  numpy as np
from  matplotlib import pyplot as plt


def load_data():
    data = np.loadtxt('data/sample.txt', delimiter=',')
    return data


class KMean(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.convergence_step = 0
        self.ori_center = np.zeros((n_clusters, 2))

    def initcenter(self, ds):
        """
        :param ds: 待聚类数据集
        :return:聚类中心
        """
        n_samples, n_features = ds.shape
        center_point = np.zeros((self.n_clusters, 2))
        for i in range(self.n_clusters):
            rnum = np.random.randint(0, n_samples)
            center_point[i, :] = ds[rnum, :]
        return center_point

    def calcdist(self, v1, v2):
        """
        :param v1: 向量1
        :param v2: 向量1
        :return:距离
        """
        return np.sqrt(sum(np.power(v1 - v2, 2)))

    def fit(self, ds):
        """
        :param ds: 待聚类数据集
        :return:none
        """
        n_samples, n_features = ds.shape
        resarray = np.zeros((n_samples, 2))
        center_point = self.initcenter(ds)
        self.ori_center = copy.deepcopy(center_point[:, :])
        centerchanged = True
        while centerchanged:
            self.convergence_step += 1
            centerchanged = False
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    #计算每个点到中心点的距离
                    dist = self.calcdist(ds[i, :],center_point[j, :])
                    if j == 0:
                        cluser_id, mindist = j, dist
                    else:
                        if dist < mindist:
                            cluser_id, mindist = j, dist
                #如果中心点发生变化，则说明未收敛，需要继续迭代
                if resarray[i, 0] != cluser_id:
                    centerchanged = True
                    resarray[i, :] = (cluser_id, mindist)
            #所有点都完成一次遍历后，更新中心点
            for j in range(self.n_clusters):
                points_in_cluster = ds[resarray[:, 0] == j]
                center_point[j, :] = np.mean(points_in_cluster, axis=0)
            self.showcluster(ds, center_point, resarray)
        print('analyze over!')
        return center_point,resarray

    def showcluster(self, ds, center_point, resarray):
        """
        :param ds: 数据集
        :param self.n_clusters: 聚类数
        :param center_point: 中心点
        :param resarray: 分类及距离
        :return:None
        """
        n_sample = ds.shape[0]
        mark = ['.r', '.g', '.b', '.k']

        if self.n_clusters > len(mark):
            print('sorry,n_clusters is too big for me!')
            return 1
        # print self.ori_center
        # print center_point
        plt.figure()
        #画所有点
        for i in range(n_sample):
            cluster_index =int(resarray[i,0])
            plt.plot(ds[i, 0], ds[i, 1], mark[cluster_index])

        #画中心点
        mark1 = ['r','g','b','k']
        for j in range(self.n_clusters):
            plt.scatter(center_point[j, 0], center_point[j, 1], marker = '*', color=mark1[j], s=250)
            plt.scatter(self.ori_center[j, 0], self.ori_center[j, 1], marker = 'D', color=mark1[j])
        plt.show()

if __name__ == "__main__" :
    ds = load_data()
    mysample = KMean(n_clusters=4)
    center_point, resarray = mysample.fit(ds)
    #print center_point,resarray
