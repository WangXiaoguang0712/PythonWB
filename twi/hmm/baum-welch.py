# _*_ coding:utf-8 _*_
import numpy as np

__author__ = 'T'

class HMM():
    def __init__(self, o, ma, mb):
        self.o = o
        np.flatiter
        self.pai = np.ones(ma.shape[0]) / ma.shape[0]
        self.ma = ma
        self.mb = mb

    def forward(self):
        alpha = np.zeros((self.ma.shape[0],len(self.o)))
        for i in range(len(self.o)):
            if i == 0:
                alpha[:, i] = self.pai * self.mb[:,self.o[i]]
            else:
                alpha[:, i] = alpha[:, i - 1].T.dot(self.ma) * self.mb[:, self.o[i]]
                # alpha[:, i] = self.ma.dot(alpha[:, i - 1] * self.mb[:,self.o[i]])
        return alpha

    def backward(self):
        beta = np.zeros((self.ma.shape[0], len(self.o)))
        for i in reversed(range(len(self.o))):
            if i == len(self.o) - 1:
                beta[:, i] = np.ones(self.ma.shape[0])
            else:
                beta[:, i] = self.ma.dot(beta[:, i + 1] * self.mb[:, self.o[i + 1]])
        return beta

    def _e_proc_(self):
        zeta = np.zeros((self.ma.shape[0], self.mb.shape[0] , len(self.o)))
        alpha = self.forward()
        beta = self.backward()
        for t in range(len(self.o) - 1):
            denominator = np.sum(alpha[:, t] * beta[:, t])
            # B*beta
            beta_tmp = self.mb[:, self.o[t + 1]].reshape(1, -1) * beta[:, t + 1].reshape(1, -1)
            # alpha * (B*beta) * A = (2x1)(1x2)(2x2)
            tmp = np.dot(alpha[:, t].reshape(-1, 1), beta_tmp) * self.ma
            zeta[:, :, t] = tmp / denominator
        gama = np.sum(zeta, axis=1)
        #print(zeta)
        #print(alpha)
        #print(beta)


    def _m_proc(self):
        pass

    def fit(self):
        self._e_proc_()

    @staticmethod
    def dict2matrix(data, dim1, dim2):
        mat = np.zeros((len(dim1), len(dim2)))
        for row in data:
            for col,val in data[row].items():
                mat[dim1.index(row)][dim2.index(col)] = val
        return mat


if __name__ == '__main__':
    status = ['like', 'dislike']
    observations = ["Coquetry", "play phone", "friendly", "leave message"]
    A = {"like": {"like": 0.5, "dislike": 0.5}, "dislike": {"like": 0.5, "dislike": 0.5}}
    B = {"like": {"Coquetry": 0.4, "play phone": 0.1, "friendly": 0.3, "leave message": 0.2},
       "dislike": {"Coquetry": 0.1, "play phone": 0.5, "friendly": 0.2, "leave message": 0.2}}
    o = [1, 2, 0, 2, 3, 0]
    A = HMM.dict2matrix(A, status, status)
    B = HMM.dict2matrix(B, status, observations)
    h = HMM(o, A, B)
    h.fit()

