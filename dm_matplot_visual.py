#coding:utf-8
__author__ = 'T'

import numpy as np
from numpy.linalg import cholesky
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def plot_gauss():
    #正态分布实例
    sample = 1000
    #一维正太分布
    #等效

    mu = 3
    sigma = 0.1

    #高斯分布数列
    np.random.seed(0)
    s = np.random.normal(mu,sigma,sample)
    plt.subplot(141)
    plt.hist(s,234,normed=True)

    np.random.seed(0)
    s = sigma*np.random.randn(sample)+mu
    plt.subplot(142)
    plt.hist(s,10,normed=True)


    np.random.seed(0)
    s = sigma*np.random.standard_normal(sample)+mu
    plt.subplot(143)
    plt.hist(s,10,normed=True)

    #二维正态分布
    mu = np.array([[1,5]])
    sigma = np.array([[1,0.5],[1.5,3]])
    R = cholesky(sigma)
    s = np.dot(np.random.randn(sample, 2),R)+mu
    plt.subplot(144)
    plt.plot(s[:,0],s[:,1],'+')

    plt.show()


def plot_special():
    # 手动指定子图句柄
    # 1.先创建一个画布
    fig = plt.figure()

    # 2.然后创建图形矩阵
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)

    ax1.hist(np.random.randn(100), bins = 20, color = 'k', alpha = 0.3)
    ax2.scatter(np.arange(30), np.arange(30)+3*np.random.randn(30))
    ax3.plot(np.random.randn(50).cumsum(),'k--')
    plt.grid(True,color='g')
    plt.show()

#给图添加文字
def plot_best():
    plt.figure(figsize=(8,6), dpi=80)
    plt.subplot(111)
    X = np.linspace(-np.pi,np.pi,256,endpoint=True)

    C,S = np.cos(X),np.sin(X)
    plt.plot(X,C,color='b',lw=1.0,linestyle='-',label='cosine')
    plt.plot(X,S,color='r',lw=2.0,linestyle='-',label='sine')
    plt.legend(loc='upper left')
    plt.axis([-4,4,-1.2,1.2])
    plt.xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],
               [r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'])
    plt.yticks([-1,0,1],
               [r'$-1$',r'$0$',r'$1$'])
    plt.grid(True)
    #
    plt.title('Hsitgram of normal')
    plt.text(-np.pi,0,r'$\mu=100,\ \sigma=15$')
    plt.xlabel('Smarts')
    plt.ylabel('Probility')
    #给特殊点做注释
    t = 2*np.pi/3
    plt.plot([t,t],[0,np.sin(t)], color ='red', linewidth=2.5, linestyle="--")
    plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',xy=(t,np.sin(t)),xycoords='data',xytext=(+10,+10),textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.plot([t,t],[0,np.cos(t)], color ='blue', linewidth=2.5, linestyle="--")
    plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',xy=(t,np.cos(t)),xycoords='data',xytext=(-90,-50),textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))


    plt.show()

def plot_log():
    plt.figure()
    x = range(1,100)
    print x
    plt.plot(x,-np.log(x),color='red')
    plt.plot(x,-np.log10(x),color='g')
    plt.show()

def plot_imshow():
    X = [[1,2],[3,4],[5,6]]
    plt.imshow(X)
    plt.show()

plot_imshow()