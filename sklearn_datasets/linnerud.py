#coding:utf-8
__author__ = 'T'
from sklearn.datasets import load_linnerud
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
名称：健身数据
使用模型：多分类
描述：20个样本
      3个特征:['Chins'(引体向上), 'Situps'(仰卧起坐), 'Jumps'（跳）]
      3个目标分类特征['Weight', 'Waist'（腰围）, 'Pulse'（脉搏）]
"""

class Linnerud():
    def __init__(self):
        ds = load_linnerud()
        self.data = ds.data
        self.n_samples,n_features = ds.data.shape
        self.target_name = ds.target_names
        self.feature_names = ds.feature_names
        print self.feature_names,self.target_name
        print ds.data

linnerud = Linnerud()


