# _*_ coding:utf-8 _*_
import numpy as np

__author__ = 'T'

class HMM():
    def __init__(self, o, ma, mb, err=0.01, iters=1000):
        self.error_min = err
        self.max_iters = iters
        self.o = o
        # np.flatiter
        self.pai = np.ones(ma.shape[0]) / ma.shape[0]
        # self.pai = [0.5, 0.5]
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

    def _e_proc(self):
        zeta = np.zeros((self.ma.shape[0], self.mb.shape[0] , len(self.o) - 1))
        alpha = self.forward()
        beta = self.backward()
        for t in range(len(self.o) - 1):
            denominator = np.sum(alpha[:, t] * beta[:, t])
            # B*beta
            beta_tmp = self.mb[:, self.o[t + 1]].reshape(1, -1) * beta[:, t + 1].reshape(1, -1)
            # alpha * (B*beta) * A = (2x1)(1x2)(2x2)
            tmp = np.dot(alpha[:, t].reshape(-1, 1), beta_tmp) * self.ma
            zeta[:, :, t] = tmp / denominator
        gamma = np.sum(zeta, axis=1)
        last_pro = alpha[:, -1].reshape(-1, 1)/np.sum(alpha[:, -1])
        gamma = np.hstack((gamma, last_pro))
        # gamma[:, -1] = alpha[:, -1]/np.sum(alpha[:, -1])
        return zeta, gamma

    def _m_proc(self, zeta, gamma):
        tmp_pai = self.pai
        tmp_ma = self.ma
        tmp_mb = self.mb

        self.pai = gamma[:,0]
        # i  ->  j 的均值 / 经过i的均值:注意数组形状
        self.ma = np.sum(zeta, axis=2) / np.sum(gamma[:, : -1], axis=1).reshape(-1, 1)
        self.mb = np.zeros_like(self.mb)
        for i in range(self.mb.shape[1]):
            vector_obs = np.array([1 if i == x else 0  for x in self.o])
            self.mb[:, i] = np.sum(gamma * vector_obs, axis=1) / np.sum(gamma, axis=1)

        if np.sum(abs(tmp_pai - self.pai)) < self.error_min and np.sum(abs(tmp_ma - self.ma)) < self.error_min and np.sum(abs(tmp_mb - self.mb)) < self.error_min:
            return True
        else:
            return False

    def fit(self):
        for i in range(self.max_iters):
            zeta, gamma = self._e_proc()
            if self._m_proc(zeta, gamma):
                print('success in iter:{0}'.format(i))
                break

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
    o = [1, 0, 1, 1 ,2, 0, 1, 2, 3, 0]
    A = HMM.dict2matrix(A, status, status)
    B = HMM.dict2matrix(B, status, observations)
    h = HMM(o, A, B)
    h.fit()
    print(h.pai)
    print(h.ma)
    print(h.mb)
