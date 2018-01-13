# _*_ coding:utf-8 _*_
import numpy as np
__author__ = 'T'

"""
step1：β4(1) = 1          β4(2) = 1          β4(3) = 1
step2:  β3(1) = β4(1)*a11*b1(white) + β4(2)*a12*b2(white) + β4(3)*a13*b3(white)
        β3(2) = β4(1)*a21*b1(white) + β4(2)*a22*b2(white) + β4(3)*a23*b3(white)
        β3(3) = β4(1)*a31*b1(white) + β4(2)*a32*b2(white) + β4(3)*a33*b3(white)
step3:P(O|M) = π1*β1(1)*b1(O1) + π2*β1(2)*b2(O1) + π3*β1(3)*b3(O1)
"""

def backward(o, seto, ma, mb, pai):
    # initialization
    l = []
    beta = np.ones((3,1))
    l.append(beta)
    for i in range(len(o) - 1):
        ep = mb[:, seto.index(o[-i - 1])].reshape(-1, 1)
        beta = A.dot(ep * beta)
        l.append(beta)
        print beta.reshape(1,-1)
    # 计算结束得到在t=1时的bata，即，生成O（2）O（3）O（4）的概率
    # 若要得到总概率，需要乘以 t=1 的生成概率，以及转移概率
    return (beta * pai *(mb[:, seto.index(o[0])].reshape(-1, 1)))

if __name__ == "__main__":
    # initialization
    s = [1, 2, 3]
    seto = ['red','white']
    o = ['red', 'white', 'red', 'white']
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    pai = np.array([[0.2, 0.4, 0.4]]).T
    r = backward(o, seto, A, B, pai)
    print np.sum(r)
