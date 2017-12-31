#coding:utf-8
__author__ = 'T'

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import  load_boston #波士顿房价数据集
from scipy.stats import pearsonr
"""
数据集名称：波士顿房价
适用模型：回归
描述：
    506个样本：
    13个特征：
    ['CRIM'(犯罪率) 'ZN'(住宅的比例) 'INDUS'(非零售营业面积比例) 'CHAS'(查尔斯河哑变量) 'NOX'(氮氧化物的浓度)
            'RM'(房间数量) 'AGE'(1940以前建造的自用单位比例) 'DIS'(加权距离) 'RAD'(公路可达性指数) 'TAX'(全额财产税税率) 'PTRATIO'(城镇师生比例) 'B'(黑人的比例) 'LSTAT'(人口的较低比例)]
            MEDV  (业主自用房屋的中位数)
    target:连续值
备注：分析房价与13个特征的关系
"""

#经典的用于回归任务的数据集
class Boston(object):
    def __init__(self):
        ds = load_boston()
        self.data = ds['data']
        self.target = ds['target']
        self.features = ds['feature_names']
        self.n_samples,self.n_features = self.data.shape#506 13

    def showdatachart(self,issort):
        #皮尔森系数
        pearsonr_list = []
        for i in range(self.n_features):
            pearsonr_list.append(pearsonr(self.data[:,i],self.target)[0])
        fea_pear = zip(self.features,pearsonr_list)
        #随机逻辑森林
        from sklearn.model_selection import cross_val_score,ShuffleSplit
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=20,max_depth=4)
        scores_list = []
        for i in range(self.n_features):
            score = cross_val_score(rf,self.data[:,i:i+1],self.target,cv=5)
            scores_list.append(round(np.mean(score),3))
        fea_score = zip(self.features,scores_list)

        xindex = [x+1 for x in range(self.n_features)]
        plt.figure(1)
        #不排序
        if issort == 0:
            plt.subplot(111)
            plt.bar(xindex,pearsonr_list,width=0.35,facecolor='lightskyblue',edgecolor='w')
            plt.bar([x+0.36 for x in xindex],scores_list,width=0.35,facecolor='r',edgecolor='w')
            for x,y in zip(xindex,pearsonr_list):
                plt.text(x,y,'%.2f' % y,ha='center',va='bottom')
            for x,y in zip(xindex,scores_list):
                plt.text(x+0.36,y,'%.2f' % y,ha='center',va='bottom')
            plt.xticks(xindex,self.features)
            plt.show()
        #排序
        else:
            fea_pear = sorted(zip(self.features,pearsonr_list),key=lambda x:x[1],reverse=True)
            fea_score = sorted(zip(self.features,scores_list),key=lambda x:x[1],reverse=True)

            pearsonr_list = sorted(pearsonr_list,reverse=True)
            scores_list = sorted(scores_list,reverse=True)

            plt.subplot(121)
            plt.bar(xindex,pearsonr_list,width=0.35,facecolor='lightskyblue',edgecolor='w')
            for x,y in zip(xindex,pearsonr_list):
                plt.text(x,y,'%.2f' % y,ha='center',va='bottom')
            plt.xticks(xindex,[x[0] for x in fea_pear])
            plt.title('sorted')

            plt.subplot(122)
            plt.bar([x for x in xindex],scores_list,width=0.35,facecolor='green',edgecolor='w')
            for x,y in zip(xindex,scores_list):
                plt.text(x,y,'%.2f' % y,ha='center',va='bottom')
            plt.xticks(xindex,[x[0] for x in fea_score])
            plt.title('sorted')
            plt.show()

boston = Boston()
boston.showdatachart(1)


