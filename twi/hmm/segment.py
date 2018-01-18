# _*_ coding:utf-8 _*_
from twi.hmm.baumwelch import HMM
from twi.hmm.viterbi import viterbi
__author__ = 'T'


if __name__ == '__main__':
    status = ['B', 'M','E','S']
    observations = ["我", '们', "在", "博", "物", "馆"]
    A = {"B": {"B": 0, "M": 0.7, "E": 0.3, "S": 0},
         "M": {"B": 0, "M": 0.5, "E": 0.5, "S": 0},
         "E": {"B": 0.5, "M": 0, "E": 0, "S": 0.5},
         "S": {"B": 0.5, "M": 0, "E": 0, "S": 0.5}}
    B = {"B": {"我": 0.6, "们": 0., "在": 0.2, "博": 0.3, "物": 0., "馆": 0.},
         "M": {"我": 0.1, "们": 0.1, "在": 0.1, "博": 0.1, "物": 0.5, "馆": 0.1},
         "E": {"我": 0.1, "们": 0.3, "在": 0.1, "博": 0.1, "物": 0.1, "馆": 0.3},
         "S": {"我": 0.2, "们": 0., "在": 0.4, "博": 0.1, "物": 0.2, "馆": 0.1}}
    o = [0,1,2,3,4,5]
    A = HMM.dict2matrix(A, status, status)
    B = HMM.dict2matrix(B, status, observations)
    h = HMM(o, A, B)
    h.fit()
    print(h.pai)
    print(h.ma)
    print(h.mb)
    viterbi(status, o, h.pai, h.ma, h.mb)
