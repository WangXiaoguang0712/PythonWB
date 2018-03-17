# _*_ coding:utf-8 _*_

import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn import datasets
from sklearn import preprocessing
from mlxtend.plotting import plot_decision_regions
# date:2018-01-22 16:58
__author__ = 'T'

class ANN_Multi():
    def __init__(self, topo_stru=[2,2,1], n_iter=20000, epsilon=0.02):
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

    def fit(self, X, y, func):
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
            for i in range(len(self.topo_stru) - 2):
                self.l_res.append(self.func(np.dot(self.l_res[-1], self.l_weight[i])))

            # calc the output
            output = np.dot(self.l_res[-1], self.l_weight[-1])
            prob = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
            self.l_res.append(prob)
            # back propagation
            # calc err and delta for each layser
            for i in range(1, len(self.topo_stru)):
                if i == 1:  # final err
                    l_err = self.l_res[-i] - y  #
                    if (n % 2000) == 0:
                        print "Error:" + str(np.sum(l_err ** 2))
                        # print 'the accuracy of this model is {0}'.format(np.sum(map(lambda x: 1 if x < 0.001 else 0,l_err))*1.0/y.shape[0])
                else:
                    l_err = l_err.dot(self.l_weight[-i + 1].T) * self.func(self.l_res[-i],True)
                delta = self.l_res[-i - 1].T.dot(l_err)  # derivatives of loss function
                delta = -1 * self.epsilon * delta
                self.l_weight[-i] += delta
        # print result
        #for i, item in enumerate(X):
        #    print item,y[i],'==>',self.l_res[-1][i]

    def predict(self, X):
        X = np.concatenate((X,np.ones((X.shape[0], 1))),axis=1)
        # calc result of each layer except the last layer
        for i in range(len(self.topo_stru) - 1):
            if i == 0:
                self.l_res.append(self.func(np.dot(X, self.l_weight[i])))
            else:
                if i < len(self.topo_stru) - 2:
                    self.l_res.append(self.func(np.dot(self.l_res[-1], self.l_weight[i])))
                else:
                    self.l_res.append(np.dot(self.l_res[-1], self.l_weight[i]))

        output = np.exp(self.l_res[-1])
        prob = output / np.sum(output, axis=1, keepdims=True)
        return np.argmax(prob, axis=1).reshape(-1,1)

    def test(self, X, y):
        y_pre = self.predict(X)
        tmp = [ 1 if np.argmax(y[i]) == y_pre[i] else 0 for i in xrange(y.shape[0])]
        print(sum(tmp) * 100.0 / len(tmp))

if __name__ == '__main__':
    np.random.seed(0)
    ohe = preprocessing.OneHotEncoder()
    ohe.fit([[1],[0]])
    X,y = datasets.make_moons(n_samples=100, noise=0.2)
    y = ohe.transform(y.reshape(-1, 1)).toarray()
    model = ANN_Multi([2,3,2])
    model.fit(X[:80],y[:80], ANN_Multi.active_simoid)
    # model.plot_decision_boundary(X,y)
    model.test(X[80:], y[80:])