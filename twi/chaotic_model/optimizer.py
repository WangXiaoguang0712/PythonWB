# _*_ coding:utf-8 _*_
__author__ = 'T'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import  Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

def test_kfold(X_train, y_train):
    # 流水线
    pipe_lr = Pipeline([('scl', StandardScaler()),
                            ('pca', PCA(n_components=2)),
                            ('clf', LogisticRegression(random_state=1))])
    skf = StratifiedKFold(n_splits=10, random_state=1)
    kfold = skf.split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        # print('Fold:%s, class dist: %s ,ACC %.3f' % (k+1, np.bincount(y_train[train]), score))
    print('CV accuracy:%.3f +/- %.3f '% (np.mean(scores), np.std(scores)))

    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
    print('cv accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def test_learning_curve(X_train, y_train):
    # 流水线
    pipe_lr = Pipeline([('scl', StandardScaler()),
                            ('pca', PCA(n_components=2)),
                            ('clf', LogisticRegression(random_state=1, penalty='l2'))])
    train_size, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10),cv=10, n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_size, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_size, train_mean + train_std, train_mean - train_std, alpha=0.2, color='blue')
    plt.plot(train_size, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
    plt.fill_between(train_size, test_mean + test_std, test_mean - test_std, alpha=0.2, color='green')
    plt.grid(True)
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.ylim([0.5, 1])
    plt.show()


def test_validation_curve(X_train, y_train):
    from sklearn.model_selection import validation_curve
    # 流水线
    pipe_lr = Pipeline([('scl', StandardScaler()),
                            ('pca', PCA(n_components=2)),
                            ('clf', LogisticRegression(random_state=1, penalty='l2'))])
    param_range = [0.001, 0.01, 0.1, 1.0, 10, 100.0]
    train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train,param_name='clf__C',
                                                 param_range=param_range, cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean,  color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.2, color='blue')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.2, color='green')
    plt.grid(True)
    plt.xscale('log')
    plt.xlabel('clf__C value')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.ylim([0.8, 1])
    plt.show()

if __name__ == "__main__":
    # 读取数据
    # df = pd.read_csv('data/breast_cancer.csv', header=None)
    df = load_breast_cancer()
    X = df['data']
    y = df['target']
    le = LabelEncoder()
    y = le.fit_transform(y)
    # 分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # K折交叉验证
    # test_kfold(X_train, y_train)
    # 学习曲线
    # test_learning_curve(X_train, y_train)
    # 验证曲线
    test_validation_curve(X_train, y_train)