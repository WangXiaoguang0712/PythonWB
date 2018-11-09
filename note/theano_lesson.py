# _*_ coding:utf-8 _*_

import numpy as np
import time
import theano
from theano import tensor as t

def tm(fn):
    def _wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        return res
    return _wrapper

#  装饰器
def fn_timer(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        res = fn(*args, **kwargs)
        print('{0} cost time {1}'.format(fn.__name__, time.clock() - start))
        return res
    return _wrapper


@fn_timer
def example1():
    # initialize
    x1 = t.scalar()
    w1 = t.scalar()
    w0 = t.scalar()
    z1 = w1 * x1 + w0
    # compile
    net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)
    # execute
    print('Net input:%2.f ' % net_input(2.0, 1.0, 0.5))


@fn_timer
def example2():
    # define
    X = t.fmatrix(name='x')
    X_sum = t.sum(X, axis=0)
    # compile
    calc_sum = theano.function(inputs=[X], outputs=X_sum)
    # execute
    arr = [[1, 2, 3], [4, 5, 6]]
    print('column sum:', calc_sum(arr))


@fn_timer
def example3():
    X = t.fmatrix('X')
    # w = t.fmatrix('w')
    w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], dtype=theano.config.floatX), name='w')
    z = t.dot(X, w.T)
    #z = X + w
    update = [[w, w + 1.0]]
    # compile
    net_input = theano.function(inputs=[X], updates=update, outputs=z)
    # execute
    data = np.asarray([[1, 2, 3]], dtype=theano.config.floatX)
    for i in range(5):
        print('z%d:' % i, net_input(data))


def example4():
    X_train = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]], dtype=theano.config.floatX)
    y_train = np.asarray([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype=theano.config.floatX)
    def train_linreg(X_train, y_train, eta, epoch):
        costs = []
        eta0 = t.fscalar(name='eta0')
        X = t.fmatrix(name='X')
        y = t.fvector(name='y')
        w = theano.shared(np.zeros(shape=(X_train.shape[1] + 1), dtype=theano.config.floatX), name='w')
        net_input = t.dot(X, w[1:]) + w[0]
        errors = y - net_input
        cost = t.sum(t.pow(errors, 2))
        # perform gradient update
        gradient = t.grad(cost, wrt=w)
        update = [(w, w - eta0 * gradient)]
        # compile
        train = theano.function(inputs=[eta0], updates=update, outputs=cost, givens={X: X_train, y:y_train,})

        for _ in range(epoch):
            costs.append(train(eta))
        return costs, w
    import matplotlib.pyplot as plt
    costs, w = train_linreg(X_train, y_train, eta=0.001, epoch=10)
    plt.plot(range(1, len(costs) + 1), costs)
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()

    def predict_linreg(X, w):
        Xt = t.fmatrix(name='Xt')
        net_input = t.dot(Xt, w[1:]) + w[0]
        predict = theano.function(inputs=[Xt], outputs=net_input, givens={w: w})
        return predict(X)

    plt.scatter(X_train, y_train, marker='s', s = 50)
    plt.plot(range(X_train.shape[0]), predict_linreg(X_train, w), color='gray', marker='o', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    example4()