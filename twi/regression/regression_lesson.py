# _*_ coding:utf-8 _*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from twi.regression.regression_linear_GD import LinearRegressionGD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def load_data():
    df = pd.read_csv('data/housing.csv', header=None, sep='\s+')
    df.columns = np.concatenate((load_boston().feature_names, ['MEDV']))
    return df

def test_sandiatu():
    '''
    散点图测试
    :return:
    '''
    df = load_data()
    sns.set(style='whitegrid', context='notebook')
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    sns.pairplot(df[cols], size=2)
    plt.show()

def test_heatmap():
    df = load_data()
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, annot=True, square=True, fmt='.2f',
                     annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
    plt.show()

def test_regress():
    '''
    回归测试
    :return:
    '''
    sc = StandardScaler()
    lr = LinearRegressionGD(n_iter=10)
    df = load_data()
    X = df[['RM']].values
    y = df[['MEDV']].values
    X_std = sc.fit_transform(X)
    y_std = sc.fit_transform(y)
    lr.fit(X_std, y_std)

    plt.plot(range(1, lr.n_iter + 1), lr.cost_)
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.show()

def test_residual():
    '''
    残差值测试：反应预测值与残差之间的关系
    :return:
    '''
    df = load_data()
    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    slr = LinearRegression()
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)

    plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Traing data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predict value')
    plt.ylabel('residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, colors='red')
    plt.xlim([-10, 50])
    plt.show()

def test_regulation():
    '''
    正则化测试：含 岭回归、拉索、弹性网络
    :return:
    '''
    df = load_data()
    X = df['RM'].values.reshape(-1, 1)
    y = df['MEDV'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    lr = LinearRegression()
    rd = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1)
    elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
    lr.fit(X_train, y_train)
    rd.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    elastic.fit(X_train, y_train)

    plt.scatter(X_train, y_train, c='blue', marker='o')
    plt.plot(X_train, lr.predict(X_train), label='lr')
    plt.plot(X_train, rd.predict(X_train), label='ridge')
    plt.plot(X_train, lasso.predict(X_train), label='lasso')
    plt.plot(X_train, elastic.predict(X_train), label='elastic')
    plt.legend(loc='upper left')
    plt.show()

def test_polynomial():
    """
    测试多项式归回
    :return:
    """
    # 加载数据
    df = load_data()
    X = df[['LSTAT']].values
    y = df[['MEDV']].values
    lr = LinearRegression()

    # 创建多项式
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # 测试数据
    X_test = np.arange(X.min(), X.max())[:, np.newaxis]

    # linear fit
    lr.fit(X, y)
    y_line_test = lr.predict(X_test)
    print(X_test.shape, y_line_test.shape)
    r2_linear = r2_score(y, lr.predict(X))

    # quadratic fit
    lr.fit(X_quad, y)
    y_quad_test = lr.predict(quadratic.fit_transform(X_test))
    r2_quad = r2_score(y, lr.predict(X_quad))

    # cubic fit
    lr.fit(X_cubic, y)
    y_cubic_test = lr.predict(cubic.fit_transform(X_test))
    r2_cubic = r2_score(y, lr.predict(X_cubic))

    # 画图
    plt.scatter(X, y, label = 'training point')
    plt.plot(X_test, y_line_test, label='linear_fit, r^2:%.2f' % r2_linear, linestyle='--')
    plt.plot(X_test, y_quad_test, label='quadratic fit r^2:%.2f' % r2_quad, color='red')
    plt.plot(X_test, y_cubic_test, label='cubic fit r^2:%.2f' % r2_cubic, color='green')
    plt.legend(loc='upper left')
    plt.show()

    # 评价
    #print('Traing MSE linear: %.3f, quadratic: %.3f' % (mean_squared_error(y, y_line_pred),
    #                                                    mean_squared_error(y, y_quad_pred)))
    print('Traing R^2 linear: %.3f, quadratic: %.3f, cubic:%.3f' % (r2_linear,
                                                        r2_quad,
                                                        r2_cubic))


if __name__ == "__main__":
    test_polynomial()