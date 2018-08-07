# _*_ coding:utf-8 _*_
__author__ = 'T'

import numpy as np
import pandas as pd
from matplotlib import  pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from twi.com.utility import plot_decision_regions

def test_bagging(X, y, X_test, y_test):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    bag = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1)
    tree = tree.fit(X, y)
    y_train_pred = tree.predict(X)
    score_tree_train = accuracy_score(y, y_train_pred)
    y_test_pred = tree.predict(X_test)
    score_tree_test = accuracy_score(y_test,y_test_pred)
    bag = bag.fit(X, y)
    y_train_pred = bag.predict(X)
    score_bag_train = accuracy_score(y, y_train_pred)
    y_test_pred = bag.predict(X_test)
    score_bag_test = accuracy_score(y_test,y_test_pred)
    print('Decision Tree Train/Test Accuracis %.3f/%.3f' % (score_tree_train, score_tree_test))
    print('Bagging Train/Test Accuracis %.3f/%.3f' % (score_bag_train, score_bag_test))
    plot_decision_regions(X, y, tree)
    plt.show()
    plot_decision_regions(X, y, bag)
    plt.show()


def test_adaboost(X, y, X_test, y_test):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=None, n_jobs=1, random_state=1)
    tree = tree.fit(X, y)
    y_train_pred = tree.predict(X)
    score_tree_train = accuracy_score(y, y_train_pred)
    y_test_pred = tree.predict(X_test)
    score_tree_test = accuracy_score(y_test,y_test_pred)
    bag = bag.fit(X, y)
    y_train_pred = bag.predict(X)
    score_bag_train = accuracy_score(y, y_train_pred)
    y_test_pred = bag.predict(X_test)
    score_bag_test = accuracy_score(y_test,y_test_pred)
    print('Decision Tree Train/Test Accuracis %.3f/%.3f' % (score_tree_train, score_tree_test))
    print('Adaboot Train/Test Accuracis %.3f/%.3f' % (score_bag_train, score_bag_test))
    plot_decision_regions(X, y, tree)
    plt.show()
    plot_decision_regions(X, y, bag)
    plt.show()

if __name__ == "__main__":
    df_wine = pd.read_csv(r'E:\PythonWB\twi\chaotic_model\data\wine.csv', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Mlic Acid', 'Ash', 'Alcalinity of ash', 'Magesium',
                       'Total phenols', 'Flavanoids', 'NonFlavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD80', 'Proline']
    df_wine = df_wine[df_wine['Class label'] != 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'Hue']].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    test_bagging(X_train, y_train, X_test, y_test)  # 测试 Bagging 集成方法
    # test_adaboost(X_train, y_train, X_test, y_test)  # 测试Adaboost 集成方法

