#coding:utf-8
__author__ = 'T'
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_files

"""
名称：手写阿拉伯数字数据集
适用模型：神经网络
描述：
    1797个样本
    64个特征：8*8数组展开后
    9个分类：0，1，2，3，4，5，6，7，8，9
备注：
images：三维数组，含1797个手写图片，每个对象保存8*8的图像，里面的元素是float64类型,[8,8]二维数组
data：将images按行展开成一行
"""

class Digits(object):
    def __init__(self):
        ds = load_digits()
        print(ds['target_names'])
        self.img = ds.images#保存8*8的图像，里面的元素是float64类型,[8,8]二维数组
        self.data = ds['data']#将images按行展开成一行，共有1797行
        self.target = ds['target']
        self.target_names = ds['target_names']#[0-9]
        self.n_samples,self.n_features = self.data.shape #(1797L, 64L)
        #ds.images.shape(1797L, 8L, 8L)


    def showdata(self):
        plt.figure()
        plt.imshow(self.img[12])
        plt.show()

dg = Digits()
dg.showdata()