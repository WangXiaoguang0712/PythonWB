# _*_ coding:utf-8 _*_
import numpy as np
from scipy.special import comb,perm
__author__ = 'T'
# 抛硬币实例
"""
实验：抛硬币，5组，每组10次，正面朝上的次数分别是5,9,8,4,7，求A,B 两枚硬币正面朝上的概率分别是多少
变量X：A,B 两枚硬币正面朝上的概率分别是多少（观测变量）
变量Z：抛的硬币是A的概率（隐变量）
"""
class EM():
    def __init__(self, theta_a, theta_b):
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.cnt = 5
        self.width = 10
        self.exams = [5,9,8,4,7]
        self.res = np.zeros((5,4))  # AH,AT BH BT

    def _e_proc(self):
        # 依据固定 theta 求Q(z)
        for idx,x in enumerate(self.exams):
            p_a = comb(self.width, x) * (self.theta_a ** x) * ((1- self.theta_a) ** (self.width - x))
            p_b = comb(self.width, x) * (self.theta_b ** x) * ((1- self.theta_b) ** (self.width - x))
            # Q(z)
            q_a = p_a / (p_a + p_b)  # 归一化
            q_b = 1 - q_a
            self.res[idx][0] = q_a * x
            self.res[idx][1] = q_a * (self.width - x)
            self.res[idx][2] = q_b * x
            self.res[idx][3] = q_b * (self.width - x)

    def _m_proc(self):
        temp_a, temp_b = self.theta_a, self.theta_b
        # 依据Q(z) 求 theate
        self.theta_a = np.sum(self.res[:, 0]) / np.sum(self.res[:, 0:2])
        self.theta_b = np.sum(self.res[:, 2]) / np.sum(self.res[:, 2:4])
        err = (temp_a - self.theta_a) ** 2 + (temp_b - self.theta_b) ** 2
        return err

    def learn(self):
        for i in range(1000):
            self._e_proc()
            err = self._m_proc()
            if err < 0.000001:
                print(i)
                break

        return self.theta_a,self.theta_b


if __name__ == '__main__':
    em = EM(0.8,0.6)
    a, b = em.learn()
    print(a)
    print(b)
