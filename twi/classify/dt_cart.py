# _*_ coding:utf-8 _*_
import numpy as np
from itertools import combinations
import copy
import pickle
from sklearn.model_selection import train_test_split

__author__ = 'T'

def load_data(filename, token='\t'):
    l = []
    with open(filename) as fo:
        for line in fo.readlines():
            l_sub = []
            for field in line.strip().split(token):
                l_sub.append(field)
            l.append(l_sub)
    return l[0], l[1:]


class dt_cart():
    def __init__(self, pl_state, pl_feature, tree_type='classify'):
        if tree_type != 'classify' and tree_type != 'regress':
            raise ValueError('incorrect tree_type parameter')
        else:
            self.tt = tree_type
        self.pk_path = 'data/btree.pkl'
        self.l_state = pl_state
        self.l_features = pl_feature
        self.num_train = 0

    def bisplit_set(self, fv):
        l = []
        for i in range(1, len(fv)):
            l.extend(list(combinations(fv, i)))
        len_l = len(l)
        return zip(l[0: len_l / 2], l[len_l - 1: (len_l / 2) - 1: -1])

    # 离散值基尼不纯度
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
        classes = sorted(set(l_classes), key=lambda x: x)
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

    # 离散值标准差
    def calc_std_discrete(self, l_feature, l_val):
        com_final = ()
        gain_final = np.inf
        features = set(l_feature)
        for com in self.bisplit_set(features):
            l_data = [[], []]
            for i in range(len(l_val)):
                if l_feature[i] in com[0]:
                    l_data[0].append(l_val[i])
                else:
                    l_data[1].append(l_val[i])
            # 计算gini
            gain = 0
            for row in l_data:
                gain += np.std(map(float, row))
            if gain < gain_final:
                gain_final = gain
                com_final = com
        return gain_final, com_final

    # 连续值的标准差
    def calc_std_continuous(self, l_feature, l_val):
        gain_final = np.inf
        sp_final = 0
        fv = map(float, l_feature)
        l_unique_val = sorted(set(fv))
        l_split_point = [(l_unique_val[x] + l_unique_val[x + 1]) / 2 for x in range(len(l_unique_val) - 1)]
        for sp in l_split_point:
            l_data = [[], []]
            for i in range(len(fv)):
                if float(fv[i]) < sp:
                    l_data[0].append(l_val[i])
                else:
                    l_data[1].append(l_val[i])
            # 计算gini
            gain = 0
            for row in l_data:
                gain += np.std(map(float, row))
            if gain < gain_final:
                gain_final = gain
                sp_final = sp
        return gain_final, ((sp_final, '-'), (sp_final, '+'))

    def select_bestfeature(self, data, l_features, l_state):
        gain_final = np.inf
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
            if self.tt == 'classify':
                if l_state[i] == 1:
                    gini, com = self.calc_gini_continuous(l_feature, l_classes)
                else:
                    gini, com = self.calc_gini_discrete(l_feature, l_classes)
            elif self.tt == 'regress':
                if l_state[i] == 1:
                    gini, com = self.calc_std_continuous(l_feature, l_classes)
                else:
                    gini, com = self.calc_std_discrete(l_feature, l_classes)
            else:
                raise ValueError('incorrect tree_type parameter')
            if gini < gain_final:
                gain_final = gini
                com_final = com
                feature = l_features[i]
        return feature, com_final

    def get_maxprobclass(self, y):
            d_y = { x:y.count(x) for x in y }
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
         # 是否只有一个分类
        target = [x[-1] for x in data]
        if len(set(target)) == 1:
            return target[0]
        # 是否没有可供分类的特征
        if len(l_features) == 1:
            return self.get_maxprobclass(target)
        # 如果特征不能二分
        if len(l_features) == 2:
            f = [x[-2] for x in data]
            if len(set(f)) == 1:
                return self.get_maxprobclass(target)


        node, com_final = self.select_bestfeature(data, l_features, l_state)
        # print(node,com_final,l_features,data)
        btree = {node:{}}
        feature_index = l_features.index(node)
        l_features.remove(node)
        state_curr = l_state.pop(feature_index)
        l_data = self.split_data(data, com_final, feature_index, state_curr)
        for i in range(2):
            l_sub_feature = copy.deepcopy(l_features)
            l_sub_state = copy.deepcopy(l_state)
            btree[node][com_final[i]] = self.build_tree(l_data[i], l_sub_feature, l_sub_state)
        return btree

    def tag_tree(self, btree, data_train):
        d_nodes = {}  # 节点字典
        max_depth = 0
        for row in data_train:
            l_path = []
            res = self._get_predict(row, self.l_features, self.l_state, btree, l_path)
            for i in reversed(range(1, len(l_path) + 1, 2)):
                if tuple(l_path)[0: i] not in d_nodes:  # 截取节点
                    d_nodes[tuple(l_path)[0: i]] = [0, 0]
                if row[-1] == res:  # 正确
                    d_nodes[tuple(l_path)[0: i]][0] += 1
                else:  # 错误
                    d_nodes[tuple(l_path)[0: i]][1] += 1
            #  print(l_path)
        return d_nodes

    def _calc_cost(self, tmp_tagedbtree, key, val):
        _cost = 0
        #  节点err
        err_node = float(val[1]) / self.num_train
        #  子树err

        #指标alpha

        return _cost

    def _prune_node(self, subtree, btree):
        pass

    def get_refinedsubtree(self, d_tagedbtree, btree):
        l_btree = [btree]
        cost_min = 1
        key_final = ()
        # tmp_tagedbtree = copy.deepcopy(d_tagedbtree)
        for key, val in d_tagedbtree.items():
            # print(key,'-', val)

            if len(key) != 1:
                # del tmp_tagedbtree[key]
                _cost = self._calc_cost(d_tagedbtree, key, val)
                if _cost < cost_min:
                    cost_min = _cost
                    key_final = key
        print(key_final)
        del d_tagedbtree[key_final]
        prunedbtree = self._prune_node(d_tagedbtree,btree)
        l_btree.append(prunedbtree)
        return l_btree

    def prune_tree(self, btree, data_train, data_test):
        # 遍历树，并标记每个节点的准确数与错误数
        tagedbtree = self.tag_tree(btree, data_train)
        # 遍历树带标记的树，计算最优子树结合
        l_btree = self.get_refinedsubtree(tagedbtree, btree)
        # 将测试数据据代入最优子树集合，选择准确率最高的子树作为最终决策树


    def fit(self, data):
        self.num_train = len(data)
        l_features, l_state = copy.deepcopy(self.l_features), copy.deepcopy(self.l_state)
        data_tmp = copy.deepcopy(data)
        btree = self.build_tree(data_tmp, l_features, l_state)

        print(btree)
        # self.prune_tree(btree, data, '')  # 暂不实现
        with open(self.pk_path, 'wb') as f:
            pickle.dump(btree, f)

    def _get_predict(self, test_data, l_feature, l_state, btree, l_path= []):
        if type(btree) == type({}):
            tmp_key = btree.keys()[0]  # 节点
            tmp_val = btree[btree.keys()[0]]  # 两个分支组成的字典
            l_path.append(tmp_key)  # 保存遍历路径
            idx = l_feature.index(tmp_key)
            s = l_state[idx]
            val = test_data[idx]
            key = tmp_val.keys()[0]  # 第一个分支key
            if s == 1:  # 连续值
                if val < key[0]:
                    node = (key[0], '-')
                else:
                    node = (key[0], '+')
            else:  # 离散值
                if val in key:
                    node = key
                else:
                    node = tmp_val.keys()[1]
            l_path.append(node)
            btree = tmp_val[node]
            # print(btree)
            return self._get_predict(test_data, l_feature, l_state, btree, l_path)
        else:
            l_path.append(btree)
            return btree

    def predict(self, test_data):
        try:
            with open(self.pk_path, 'rb') as f:
                btree = pickle.load(f)
        except:
            raise 'please build tree first!'
        tmp = copy.deepcopy(btree)
        l_res_predict = []
        l_res_actual = []
        for row in test_data:
            l_res_actual.append(row[-1])
            res = self._get_predict(row, self.l_features, self.l_state, tmp)
            print(row[0: -1],'==>',res)

    def test_accuracy(self, test_data):
        try:
            with open(self.pk_path, 'rb') as f:
                btree = pickle.load(f)
        except:
            raise 'please build tree first!'
        tmp = copy.deepcopy(btree)
        l_res_predict = []
        for row in test_data:
            res = self._get_predict(row, self.l_features, self.l_state, tmp)
            l_res_predict.append(1 if res == row[-1] else 0)
        return float(sum(l_res_predict)) / len(l_res_predict)


if __name__ == '__main__':
    # feature, data = load_data('data/car.data', ',')
    # l_state = [0, 0, 0, 0, 0, 0]
    feature, data = load_data('data/cart2.txt', '\t')
    l_state = [0, 0, 0, 1]
    data_train, data_test = train_test_split(data, test_size=0)
    m = dt_cart(pl_feature=feature, pl_state=l_state, tree_type='regress')
    m.fit(data_train)
    # m.predict([data[-1]], feature, l_state)
    # print(m.test_accuracy(data_test))

