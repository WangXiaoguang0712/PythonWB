# _*_ coding:utf-8 _*_
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
__author__ = 'T'


class KNNClassifier():
    def __init__(self, n_neighbours=5):
        self.n_neighbours = n_neighbours
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        distance = np.sqrt(np.sum((np.tile(x, (self.X.shape[0], 1)) - self.X) ** 2, axis=1))  # 欧式距离
        idx = np.argsort(distance)  # 索引顺序
        l_label = self.y[idx[:self.n_neighbours]]
        d_label = {x:list(l_label).count(x) for x in l_label}
        d_label_sorted = sorted(d_label.items(), key=lambda x:x[1], reverse=True)
        return d_label_sorted[0][0]

    def score(self, X, y):
        k = 0
        for i in xrange(X.shape[0]):
            if y[i] == self.predict(X[i]):
                k += 1
        return k*100.0/X.shape[0]


if __name__ == "__main__":
    knn = KNNClassifier(n_neighbours=5)
    ds = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.2, shuffle=True)
    knn.fit(x_train, y_train)
    #res = knn.predict(x_test[10])
    #print(res)
    #print(y_test[10])
    res2 = knn.score(x_test, y_test)
    print(res2)