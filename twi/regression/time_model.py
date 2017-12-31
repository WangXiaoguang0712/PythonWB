#coding:utf-8
__author__ = 'T'

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Xt = 0 - 1.5Xt_1 + 0.5Xt_2
class TimeModel(object):
    def __init__(self):
        iter = 300 - 2 # 生成测试数据
        self.k = 20 # AR(p)
        self.l = [0,1]
        self.l_acf = []
        self.l_pacf = []

        ts_data = pd.read_csv('data/AirPassengers.csv', parse_dates=['Month'], index_col='Month',
                              date_parser=lambda dates:pd.datetime.strptime(dates, '%Y-%m'))

        self.l = np.array(ts_data["#Passengers"])
        """
        for i in range(iter):
            t = len(self.l)
            x = 0 + 1.5 * self.l[t - 1] - 0.5 * self.l[t - 2]
            #x = 0 + 1.1/0.98 * self.l[t - 1]
            self.l.append(x)
        """
    # 计算卷积
    def calc_rollmean_m2(self, sp):
        n = 5
        weight = np.ones(n) / n
        sp_5 = np.convolve(weight, sp)[n - 1:- n + 1]

    def show_testdata(self):
        plt.figure()
        plt.plot(range(len(self.l)),self.l,'-r')
        #plt.show()

    #自相关系数
    def calc_acf(self):
        len_l = len(self.l)
        for j in range(1,self.k+1):
            x1 = self.l[0:len_l - j]
            x2 = self.l[j:len_l]
            x1_mean,x1_std = np.mean(x1),np.std(x1)
            x2_mean,x2_std = np.mean(x2),np.std(x2)
            p = np.mean((x1 - x1_mean) * (x2 - x2_mean))/(x1_std * x2_std)
            self.l_acf.append(p)
        print self.l_acf
        from statsmodels.tsa import stattools as stt
        lag_acf = stt.acf(self.l, nlags=20)
        print lag_acf

    #偏自相关系数
    def calc_pacf(self):
        len_l = len(self.l)
        len_subl = 10
        for i in range(1,10):
            x1 = self.l[0:len_subl]
            x2 = self.l[i * len_subl:(i+1) * len_subl]
            x1_mean,x1_std = np.mean(x1),np.std(x1)
            x2_mean,x2_std = np.mean(x2),np.std(x2)
            p = np.mean((x1 - x1_mean) * (x2 - x2_mean))/(x1_std * x2_std)
            self.l_pacf.append(p)

    def show_acf_pacf(self):
        plt.figure()
        p1, = plt.plot(range(len(self.l_acf)),self.l_acf,'-g')
        p2, = plt.plot(range(len(self.l_pacf)),self.l_pacf,'-r')
        plt.grid(True)
        plt.title('acf&pacf')
        plt.xlabel('number of iter')
        plt.ylabel('value of coffient', fontsize=16)
        plt.legend([p1,p2],['acf','pacf'],loc = 'upper right')
        plt.show()

if __name__ == "__main__":
    tm = TimeModel()
    tm.show_testdata()
    tm.calc_acf()
    tm.calc_pacf()
    tm.show_acf_pacf()



