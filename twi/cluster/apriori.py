#coding:utf-8
__author__ = 'T'

import  numpy as np
from matplotlib import pyplot as plt

class Apriori(object):
    def __init__(self,support_min = 0.5,conf_min = 0.5):
        self.k = 0
        self.support_min = 0.5
        self.conf_min = 0.5
        self.support_data = {}
        self.conf_data = []
        self.ds_test = [ [ 1, 3, 4 ], [ 2, 3, 5 ], [ 1, 2, 3, 5 ], [ 2, 5 ] ]

    def is_apriori(self,set_ck,list_set_lksub1):
        """
        依据k-1项频繁集判断k项候选集是否是频繁集
        :param set_ck: k项候选集
        :param set_lksub1: k-1项频繁集
        :return:True or False
        """
        for item in set_ck:
            set_ck_sub = set_ck - frozenset([item]) #最可能不是频繁集的子集
            if set_ck_sub not in list_set_lksub1:
                return False
            return  True

    def generate_list_set_c1(self,ds):
        """
        创建1项集
        :param ds: 原数据集
        :return: 一项集
        """
        c1 = []
        for tran in ds:
            for item in tran:
                if [item] not in c1:
                    c1.append([item])
        c1.sort()
        return map(frozenset,c1)

    def generate_list_set_ck(self,list_set_l_ksub1,k):
        """
        生成k项集
        :param ds:
        :param set_l_ksub1: k-1 项频繁集
        :return:k 项候选集
        """
        list_set_ck = []
        len_set_l = len(list_set_l_ksub1)
        for i in range(len_set_l):
            for j in range(i+1,len_set_l):
                list_set_l_i = list(list_set_l_ksub1[i])
                list_set_l_j = list(list_set_l_ksub1[j])
                list_set_l_i.sort()
                list_set_l_j.sort()
                if list_set_l_i[:k - 2] == list_set_l_j[:k - 2]:
                    set_iuj = list_set_l_ksub1[i] | list_set_l_ksub1[j]
                    if self.is_apriori(set_iuj,list_set_l_ksub1):#判断是否是频繁集
                        list_set_ck.append(list_set_l_ksub1[i] | list_set_l_ksub1[j])
        return list_set_ck



    def generate_list_set_lk(self,list_set_all,list_set_ck):
        """
        生成频繁集
        :param ds:
        :param set_c: 候选集
        :return:频繁集和支持度
        """
        support_data = {}
        for item in list_set_ck:
            for tran in list_set_all:
                if item.issubset(tran):
                    support_data[item] = support_data.get(item,0) + 1
                    continue
        len_list_set_all = float(len(list_set_all))
        list_set_l = []
        for set_i in list_set_ck:
            support_data[set_i] = support_data[set_i]/len_list_set_all
            if support_data[set_i] >= self.support_min:
                self.support_data[set_i] = support_data[set_i]
                list_set_l.append(set_i)
        return list_set_l


    def analyze(self,ds):
        ds = self.ds_test if ds is None else ds
        # 'start generate lk...'
        list_l = []
        list_set_all = map(set,ds)
        k = 1
        #generate 1-item freq set
        c1 = self.generate_list_set_c1(list_set_all)
        l1 = self.generate_list_set_lk(list_set_all,c1)
        list_l.append(l1)
        #genergate k-item freq set
        while len(list_l[k - 1]) > 0:
            k += 1
            ck = self.generate_list_set_ck(list_l[k - 2],k)
            lk = self.generate_list_set_lk(list_set_all,ck)
            if len(lk) > 0:
                list_l.append(lk)
                self.k = k
            else:
                break

        # 'start generate rules...'
        len_list_l = len(list_l)
        for i in range(len_list_l):#i=0,1,2,3
            for j in range(i + 1,len_list_l):#j=i+1=1,2,3,4
                for item_i in list_l[i]:#子集
                    for item_j in list_l[j]:#超集
                        if item_i.issubset(item_j):
                            rule_i_j = (item_i,item_j - item_i,self.support_data[item_j]/self.support_data[item_i])
                            self.conf_data.append(rule_i_j)
                            #print '{0} => {1}:{2}'.format(rule_i_j[0],rule_i_j[1],rule_i_j[2])

    def showrules(self):
        for item in self.conf_data:
            print '{0} => {1}:{2}'.format(item[0],item[1],item[2])


if __name__ == "__main__":
    ap = Apriori(support_min=0.6)
    ap.analyze(None)
    print ap.k
    print ap.support_data
    ap.showrules()