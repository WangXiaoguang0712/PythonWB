# _*_ coding:utf-8 _*_
import numpy as np
import math
import copy
__author__ = 'T'

def load_dataset():
    X = np.array([[0.8, 0.5, 0.9, 0.5],
                  [0.7, 0.9, 0.5, 0.5],
                  [0.3, 0.4, 0.3, 0.5],
                  [0.2, 0.8, 0.4, 0.5],
                  [0.7, 0.6, 0.6, 0.5]])
    y = np.array([0, 1, 1, 0, 1])
    feature = ['is rich','is handsom','is height','issteady']
    return X, y, feature

class DecisionTree():
    def __init__(self, method='ID3'):
        self.method = method
        self.k = 0

    def calc_shannon_entropy(self, y):
        len_y = y.shape[0]
        d_y = {}
        for i in y:
            i = tuple([i])
            if i in d_y:
                d_y[i] += 1
            else:
                d_y[i] = 1
        entropy = 0.0
        for key,val in d_y.iteritems():
            possibility = float(val) / len_y
            entropy += - possibility * math.log(possibility, 2)
        return entropy

    def calc_condition_entropy(self, fv, y):
        len_y = y.shape[0]
        c_entropy = 0.0
        if len(set(list(fv))) == 1:
            c_entropy = self.calc_shannon_entropy(y)
            final_sp = fv[0]
        else:
            l_unique_val = sorted(list(set(list(fv))))
            l_split_point = [(l_unique_val[x] + l_unique_val[x + 1]) / 2 for x in range(len(l_unique_val) - 1)]
            final_sp = l_split_point[0]
            for sp in l_split_point:
                tmp_entropy = 0.0
                tmp_sp = sp
                for i in range(2):  # 二分裂
                    cond1 = fv >= sp
                    cond2 = i == 0
                    y_new = y[cond1 ^ cond2]  # 异或
                    tmp_entropy += float(len(y_new)) / len_y * self.calc_shannon_entropy(y_new)
                c_entropy = max(c_entropy, tmp_entropy)
                final_sp = sp if tmp_entropy > c_entropy else final_sp
        return c_entropy, final_sp

    def calc_entropy_gain(self, fv, y, isratio=False):
        c_entropy, final_sp = self.calc_condition_entropy(fv, y)
        gain = self.calc_shannon_entropy(y) - c_entropy
        if isratio:
            tmp_fv = copy.deepcopy(fv)
            tmp_fv[tmp_fv >= final_sp] = 1
            tmp_fv[tmp_fv < final_sp] = 0
            entropy_f = self.calc_shannon_entropy(tmp_fv)
            if entropy_f == 0:
                return 0, final_sp
            else:
                return gain / entropy_f, final_sp
        else:
            return gain, final_sp

    def get_maxprobclass(self, y):
        d_y = {}
        y = list(y)
        for i in y:
            if i in d_y:
                d_y[i] += 1
            else:
                d_y[i] = 1
        return sorted(d_y.iteritems(), key=lambda x: x[1], reverse=True)[0][0]

    def select_feature(self, X, y):
        l_diminish = []
        best_gain = -1
        final_feature_sp = 0
        if self.method == 'ID3':
            for i in range(X.shape[1]):
                gain, final_sp = self.calc_entropy_gain(X[:, i], y)
                l_diminish.append(gain)
                final_feature_sp = final_sp if gain > best_gain else final_feature_sp
            return np.argmax(l_diminish), final_feature_sp
        elif self.method == 'C4.5':
            for i in range(X.shape[1]):
                gain, final_sp = self.calc_entropy_gain(X[:, i], y, isratio=True)
                l_diminish.append(gain)
                final_feature_sp = final_sp if gain > best_gain else final_feature_sp
                best_gain = gain if gain > best_gain else best_gain
            return np.argmax(l_diminish), final_feature_sp
        elif self.method == 'CART':
            pass
        else:
            raise ValueError('illegal parameter!')

    def fit_tree(self, X, y, feature):
        if len(y) == 0:  # 遇到不存在的分类情况时
            return ''
        if len(set(y)) == 1:  # 分类只有一个时返回
            return y[0]
        if len(feature) == 0:  # 没有特征可以选择了，按概率算类别，并返回
            return self.get_maxprobclass(y)

        feature_index, feature_sp = self.select_feature(X, y)
        node = feature[feature_index]
        del(feature[feature_index])
        mytree = {node: {}}
        bbranch = [(str(feature_sp),'-'), (str(feature_sp),'+')]
        for item in bbranch:
            cond1 = X[:,feature_index] >= float(item[0])
            cond2 = item[1] == '-'
            p_X, p_y = X[cond1 ^ cond2], y[cond1 ^ cond2]
            p_X = np.concatenate((p_X[:,:feature_index], p_X[:,feature_index + 1:]), axis=1)
            subfeature = copy.deepcopy(feature)
            mytree[node][item] = self.fit_tree(p_X, p_y, subfeature)
        return mytree


    def fit(self,X, y, feature):
        tree = self.fit_tree(X, y, feature)
        print(tree)

    def predict(self):
        pass

if __name__ == '__main__':
    X, y, feature = load_dataset()
    m = DecisionTree(method='C4.5')
    #print m.calc_shannon_entropy(y)
    #print m.calc_condition_entropy()
    # m.select_feature(X, y)
    # print m.get_maxprobclass(y)
    m.fit(X, y, feature)