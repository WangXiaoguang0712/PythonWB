#coding:utf-8
__author__ = 'T'
from sklearn.datasets import load_diabetes #适用于线性回归
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
数据集名称：糖尿病
食用模型：多元回归
描述：
    442个样本：
    10个特征值：
        # 年龄
        # 性别
        #体质指数
        #血压
        #s1,s2,s3,s4,s4,s6  (六种血清的化验数据)
     target:连续变量

备注： 该数据集包括442个病人的生理数据及一年以后的病情发展情况。
！！！但请注意，以上的数据是经过特殊处理， 10个数据中的每个都做了均值中心化处理，然后又用标准差乘以个体数量调整了数值范围。验证就会发现任何一列的所有数值平方和为1.
"""
class Diabetes(object):
    def __init__(self):
        ds = load_diabetes()
        self.data = ds['data']
        self.target = ds['target']
        self.feature_names = ds['feature_names']
        #self.target_names = ds['target_names']
        self.n_samples,self.n_features = self.data.shape #442，10
        #print ds.DESCR
    #数据查验
    def testdata(self):
        from sklearn.model_selection import train_test_split
        from sklearn import linear_model
        x_train,x_test,y_train,y_test = train_test_split(self.data,self.target,random_state=3)

        lrg = linear_model.LinearRegression()
        lrg.fit(self.data,self.target)

        #如何评价以上的模型优劣呢？我们可以引入方差，方差越接近于1,模型越好.
        # 方差: 统计中的方差（样本方差）是各个数据分别与其平均数之差的平方的和的平均数
        print lrg.score( x_test,y_test)
        plt.figure(  figsize=(8,12))
        #matplot显示图例中的中文问题 :   https://www.zhihu.com/question/25404709/answer/67672003
        import matplotlib.font_manager as fm
        #mac中的字体问题请看: https://zhidao.baidu.com/question/161361596.html
        #myfont = fm.FontProperties(fname='/Library/Fonts/Xingkai.ttc')
        #循环10个特征
        for f in range(0,10):
            #取出测试集中第f特征列的值, 这样取出来的数组变成一维的了，
            xi_test=x_test[:,f]
            #取出训练集中第f特征列的值
            xi_train=x_train[:,f]

            #将一维数组转为二维的
            xi_test=xi_test[:,np.newaxis]
            xi_train=xi_train[:,np.newaxis]

            plt.ylabel(u'病情数值')
            lrg.fit( xi_train,y_train)   #根据第f特征列进行训练
            y=lrg.predict( xi_test )       #根据上面训练的模型进行预测,得到预测结果y

            #加入子图
            plt.subplot(5,2,f+1)   # 5表示10个图分为5行, 2表示每行2个图, f+1表示图的编号，可以使用这个编号控制这个图
            #绘制点   代表测试集的数据分布情况
            plt.scatter(  xi_test,y_test,color='k' )
            #绘制线
            plt.plot(xi_test,y,color='b',linewidth=3)

        #plt.savefig('python_1.png')
        plt.show()


dbtz = Diabetes()
dbtz.testdata()

