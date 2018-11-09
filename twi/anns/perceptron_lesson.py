#coding:utf-8
import time
import re
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

if __name__ == "__main__":
    ds = load_iris()
    #X = np.concatenate((ds.data, ds.target.reshape(-1, 1)), axis=1)
    #df = pd.DataFrame(X, index=None, columns=ds.feature_names.append('label'))
    X = ds.data[:100]
    y = ds.target[:100]

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    print(ppn.errors_)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.show()