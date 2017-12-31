#coding:utf-8
__author__ = 'T'

import numpy as np
import matplotlib.pyplot as plt
import random

class SVM(object):
    #构造函数
    def __int__(self,xdata,ydata,alpha,c=1000000,w=[0,0],b=0):
        self.c = c
        self.w = w
        self.b = b
        self.xdata = xdata
        self.ydata = ydata
        self.alpha = alpha
    #生成测试数据
    def getTestData(self,n):
        xdata = []
        for i in range(n):
            idata = [random.randint(0,20),random.randint(0,20)]
            if sum(idata)>=20:
                idata = [i + 5 for i in idata]
            if not idata in xdata:
                xdata.append(idata)
        ydata = [1 if sum(i) > 20 else -1 for i in xdata]
        return xdata,ydata
    #核函数，这里是线性可分因此就是普通的内积运算
    def kernels(self,x1,x2):
        return  x1[0]*x2[0]+x1[1]*x2[1]
    #核矩阵
    def kernelMatrix(self):
        mat = np.eye(len(self.xdata),len(self.xdata))
        for i in range(len(self.xdata)):
            for j in range(len(self.xdata)):
                mat[i,j] = self.kernels(self.xdata[i],self.xdata[j])
        return mat
    #求ui
    def ui(self,i):
        a = 0
        for j in range(len(self.xdata)):
            a = a + self.alpha[j]*self.ydata[j]*(self.xdata[j,0]*self.xdata[i,0]*self.xdata[j,1]*self.xdata[i,1])
        a = a + self.b
        return a

    # 求Ei=ui-yi
    def Ei(self,i):
        return self.ui(i) - self.ydata[i]

    # 找第二个alpha2在alpha向量中的位置，通过max|Ei-Ej|
    #def alpha2(self,i,tflist):



#svm = SVM()
a = np.eye(3,2)
print a