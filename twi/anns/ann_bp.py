# _*_ coding:utf-8 _*_

import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions
# date:2017-12-23 13:49
__author__ = 'T'


class ANN_BP():
    def __init__(self, topo_stru=[2,2,1], n_iter=20000, epsilon=0.01):
        self.topo_stru = topo_stru
        self.n_iter = n_iter
        self.l_weight = []
        self.l_res = []
        self.epsilon = epsilon  # 学习率
        self.func = None

    @staticmethod
    def active_simoid(x, derive=False):
        if derive == True:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    @staticmethod
    def active_tanh(x, derive=False):
        if derive == True:
            return 1 - x ** 2
        else:
            return np.tanh(x)

    def calc_accuracy(self):
        pass

    def fit(self,X,y,func):
        if type(X) != np.ndarray or type(y) != np.ndarray:
            raise TypeError('You shoud input correct datatype,such as np.ndarray')
        if X.shape[0] != y.shape[0] or X.ndim != y.ndim:
            raise ValueError('The shape of y is not match X')
        X = np.concatenate((X,np.ones((X.shape[0], 1))),axis=1)
        self.func = func
        self.topo_stru = [ x + 1 if i < len(self.topo_stru) - 1 else x for i,x in enumerate(self.topo_stru)]
        # init weight matrix
        for i in range(len(self.topo_stru) - 1):
            self.l_weight.append(np.random.randn(self.topo_stru[i], self.topo_stru[i + 1]) / np.sqrt(self.topo_stru[i]))
        # start iterate
        for n in range(self.n_iter):
            self.l_res = [X]  # convenient for iter
            # forward propagation
            # calc result of each layer
            for i in range(len(self.topo_stru) - 1):
                self.l_res.append(self.func(np.dot(self.l_res[-1], self.l_weight[i])))

            # back propagation
            # calc err and delta for each layser
            for i in range(1, len(self.topo_stru)):
                if i == 1:  # final err
                    l_err = (self.l_res[-i] - y) * self.func(self.l_res[-i],True)  # err * confident
                    if (n% 2000) == 0:
                        print "Error:" + str(np.sum(l_err ** 2))
                        # print 'the accuracy of this model is {0}'.format(np.sum(map(lambda x: 1 if x < 0.001 else 0,l_err))*1.0/y.shape[0])
                else:
                    l_err = l_err.dot(self.l_weight[-i + 1].T) * self.func(self.l_res[-i],True)
                delta = self.l_res[-i - 1].T.dot(l_err)  # derivatives of loss function
                delta = -1 * self.epsilon * delta
                self.l_weight[-i] += delta
        # print result
        for i,item in enumerate(X):
            print item,y[i],'==>',self.l_res[-1][i]

    def predict(self,X):
        X = np.concatenate((X,np.ones((X.shape[0], 1))),axis=1)
        # calc result of each layer
        for i in range(len(self.topo_stru) - 1):
            if i == 0:
                self.l_res.append(self.func(np.dot(X, self.l_weight[i])))
            else:
                self.l_res.append(self.func(np.dot(self.l_res[-1], self.l_weight[i])))
        return self.l_res[-1]

    def plot_decision_boundary(self, X, y):
        # 设定最大最小值，附加一点点边缘填充
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # 用预测函数预测一下
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # 然后画出图
        plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(Z.max() + 2) - 0.1, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap=plt.cm.Spectral)
        plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    X,y = datasets.make_moons(n_samples=4, noise=0.5)
    model = ANN_BP([2,3,1])
    model.fit(X,y.reshape(-1,1),ANN_BP.active_simoid)
    model.plot_decision_boundary(X,y)
    #plot_decision_regions(X,y,model,legend=0)
    #plt.title("Decision Boundary for hidden layer size %d")
    #plt.show()

    #print model.predict(X)