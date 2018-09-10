# _*_ coding:utf-8 _*_
__author__ = 'T'


import numpy as np
import pandas as pd
from twi.cluster.apriori import Apriori


def load_data():
    data_path = 'data/yiyu.csv'
    d_pat = {}
    with open(data_path, encoding='utf8', mode= 'r', errors='ignore') as f:
        for line in f.readlines():
            fields = line.replace('\\,',' ').replace('\n','').split(',')
            if d_pat.get(fields[3]) is None:
                d_pat[fields[3]] = set({})
            d_pat[fields[3]].add(fields[4])

    l_pat_med = []
    for key, val in d_pat.items():
        l_pat_med.append(val)
    return l_pat_med


def analysis():
    data = load_data()
    ap = Apriori(support_min=0.05)
    ap.analyze(data)
    ap.showrules()


def load_data2(hsp):
    data_path = 'data/yiyu.csv'
    d_pat = {}
    df_med = pd.read_csv(data_path)
    df_med.columns = ['hspcode', 'icd10', 'icdname', 'id', 'med']
    tmp = df_med[df_med.hspcode==hsp][['id','med']]
    for idx in tmp.index:
        id = tmp.loc[idx][0]
        med = tmp.loc[idx][1]
        if d_pat.get(id) is None:
                d_pat[id] = set({})
        d_pat[id].add(med)
    l_pat_med = []
    for key, val in d_pat.items():
        l_pat_med.append(val)
    return l_pat_med


def analysis2():
    ap = Apriori(support_min=0.05)
    print('**' * 20, '回龙观', '**' * 20)
    data_54X = load_data2('40068654XA')
    ap.analyze(data_54X)
    ap.showrules()
    print('**' * 20, '安定', '**' * 20)
    data_465 = load_data2('400688465A')
    ap = Apriori(support_min=0.05)
    ap.analyze(data_465)
    ap.showrules()

if __name__ == "__main__":
    analysis2()
