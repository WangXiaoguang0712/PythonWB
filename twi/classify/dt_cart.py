# _*_ coding:utf-8 _*_
import numpy as np
from itertools import combinations
import copy
import pickle

__author__ = 'T'

def load_data(filename):
    l = []
    with open(filename) as fo:
        for line in fo.readlines():
            l_sub = []
            for field in line.strip().split('\t'):
                l_sub.append(field)
            l.append(l_sub)
    return l[0], l[1:]


class dt_cart():
    def __init__(self):
        self.pk_path = 'data/bitree.pkl'

    def bisplit_set(self, fv):
        l = []
        for i in range(1, len(fv)):
            l.extend(list(combinations(fv, i)))
        len_l = len(l)
        return zip(l[0: len_l / 2], l[len_l - 1: (len_l / 2) - 1: -1])

    def calc_gini_discrete(self, l_feature, l_classes):
        com_final = ()
        gain_final = 1.1
        classes = sorted(set(l_classes), key=lambda x:x)
        features = set(l_feature)
        for com in self.bisplit_set(features):

            mat_data = np.zeros((len(com), len(classes)))
            for i in range(len(l_feature)):
                if l_feature[i] in com[0]:
                    mat_data[0][classes.index(l_classes[i])] += 1
                else:
                    mat_data[1][classes.index(l_classes[i])] += 1
            # 计算gini
            gini = 0
            for row in mat_data:
                gini_sub = 1 - np.sum((row / np.sum(row)) ** 2)
                gini += np.sum(row) / np.sum(mat_data) * gini_sub
            if gini < gain_final:
                gain_final = gini
                com_final = com
        return gain_final, com_final

    def calc_gini_continuous(self, l_feature, l_classes):
        gain_final = 1.1
        sp_final = 0
        fv = map(float, l_feature)
        l_unique_val = sorted(set(fv))
        l_split_point = [(l_unique_val[x] + l_unique_val[x + 1]) / 2 for x in range(len(l_unique_val) - 1)]
        classes = sorted(set(l_classes), key=lambda x:x)
        for sp in l_split_point:
            mat_data = np.zeros((2, len(classes)))
            for i in range(len(fv)):
                if float(fv[i]) < sp:
                    mat_data[0][classes.index(l_classes[i])] += 1
                else:
                    mat_data[1][classes.index(l_classes[i])] += 1
            # 计算gini
            gini = 0
            for row in mat_data:
                gini_sub = 1 - np.sum((row / np.sum(row)) ** 2)
                gini += np.sum(row) / np.sum(mat_data) * gini_sub
            if gini < gain_final:
                gain_final = gini
                sp_final = sp
        return gain_final, ((sp_final, '-'), (sp_final, '+'))

    def select_bestfeature(self, data, l_features, l_state):
        gini_final = 1
        com_final = ()
        feature = ''
        l_classes = []
        for i in range(len(l_features) - 1):
            l_feature = []
            for row in data:
                if i == 0:  # 第一次遍历读取分类
                    l_classes.append(row[-1])
                l_feature.append(row[i])
            # 判断是连续还是离散
            if l_state[i] == 1:
                gini, com = self.calc_gini_continuous(l_feature, l_classes)
            else:
                gini, com = self.calc_gini_discrete(l_feature, l_classes)
            if gini < gini_final:
                gini_final = gini
                com_final = com
                feature = l_features[i]
        return feature, com_final

    def get_maxprobclass(self, y):
            d_y = {}
            y = list(y)
            for i in y:
                if i in d_y:
                    d_y[i] += 1
                else:
                    d_y[i] = 1
            return sorted(d_y.iteritems(), key=lambda x: x[1], reverse=True)[0][0]

    def split_data(self, data, com_final, idx, s):
        data_1, data_2 = [], []
        if s == 1:
            for row in data:
                if float(row[idx]) < com_final[0][0]:
                    row.pop(idx)
                    data_1.append(row)
                else:
                    row.pop(idx)
                    data_2.append(row)
        else:
            for row in data:
                if row[idx] in com_final[0]:
                    row.pop(idx)
                    data_1.append(row)
                else:
                    row.pop(idx)
                    data_2.append(row)
        return [data_1, data_2]

    def build_tree(self, data, l_features, l_state):
        #  数据集是否为空
        if len(data) == 0:
            return ''
        # 是否没有可供分类的特征
        if len(l_features) == 0:
            return self.get_maxprobclass()
        # 是否只有一个分类
        target = [x[-1] for x in data]
        if len(set(target)) == 1:
            return target[0]

        node, com_final = self.select_bestfeature(data, l_features, l_state)
        bitree = {node:{}}
        feature_index = l_features.index(node)
        l_features.remove(node)
        state_curr = l_state.pop(feature_index)
        l_data = self.split_data(data, com_final, feature_index, state_curr)
        for i in range(2):
            l_sub_feature = copy.deepcopy(l_features)
            l_sub_state = copy.deepcopy(l_state)
            bitree[node][com_final[i]] = self.build_tree(l_data[i], l_sub_feature, l_sub_state)
        return bitree

    def fit(self, data, feature, l_state):
        btree = self.build_tree(data, feature, l_state)
        with open(self.pk_path, 'wb') as f:
            pickle.dump(btree, f)

    def predict(self, test_data, l_feature, l_state,):
        try:
            with open(self.pk_path, 'rb') as f:
                bitree = pickle.load(f)
        except:
            raise 'please build tree first!'
        tmp = copy.deepcopy(bitree)
        tmp = tmp[tmp.keys()[0]]
        while type(tmp) == type({}):
            l_key = tmp.keys()
            if len(l_key) == 1:
                key = l_key[0]
            else:
                for key in tmp.keys():
                    if key == l_feature.index()
            note = tmp.keys()[0]

if __name__ == '__main__':
    feature, data = load_data('data/cart1.txt')
    l_state = [0, 0, 1, 0]
    m = dt_cart()
    m.fit(data, feature, l_state)
    m.predict(data[-1], feature, l_state)
    # print(m.calc_gini_continuous(data, 2))

