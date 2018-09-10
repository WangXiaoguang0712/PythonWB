# _*_ coding:utf-8 _*_

import numpy as np

class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.n_iter = n_iter
        self.eta = eta
        self.cost_ = []

    def fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.w_ = np.zeros((1, X.shape[1]))
        for i in range(self.n_iter):
            output = self.net_input(X)
            output = self.net_input(X)
            errors = y - output
            self.w_ += self.eta * errors.T.dot(X)
            cost = np.sum((errors ** 2))
            self.cost_.append(cost)


    def net_input(self, X):
        return np.dot(X, self.w_.T)

    def predict(self, X):
        return self.net_input(X)

if __name__ == "__main__":
    pass