# _*_ coding:utf-8 _*_
import numpy as np
__author__ = 'T'
"""
 step1：α1(1) =π1*b1(red)=0.2*0.5=0.1          α1(2)=π2*b2(red)==0.4*0.4= 0.16          α1(3)=π3*b3(red)==0.4*0.7=0.21
 step2：α2(1)=α1(1)*a11*b1(white) + α1(2)*a21*b1(white) + α1(3)*a31*b1(white) = (α1(1)*a11 + α1(2)*a21 + α1(3)*a31)*b1(white)
        α2(2) = (α1(1)*a12 + α1(2)*a22 + α1(3)*a32)*b2(white)
        α2(3) = (α1(1)*a13 + α1(2)*a23 + α1(3)*a33)*b3(white)
"""

# forward-algorithm
def forward(o, seto, ma, mb, pai):
    for i in range(0, len(o)):
        if i == 0:
            alpha = pai
            temp = alpha * (mb[:, seto.index(o[i])].reshape(-1, 1))
        else:
            alpha = temp.T.dot(A)
            temp = alpha.T * (mb[:, seto.index(o[i])].reshape(-1, 1))
        print "alpha{0}:{1}".format(i,np.sum(temp))
    return np.sum(temp)

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

    r = forward(o, seto, A, B, pai)
    print r