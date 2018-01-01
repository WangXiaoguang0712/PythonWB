# _*_ coding:utf-8 _*_
import copy, numpy as np
import pickle

def simoid(x,derive=False):
    if derive == True:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

def load_data():
    n_max = 8
    n_exp_max = 2 ** n_max
    d_bin = {}
    l_tmp = np.unpackbits(np.array([range(n_exp_max)],dtype=np.uint8).T,axis=1)
    for i in range(n_exp_max):
        d_bin[i] = l_tmp[i]
    return d_bin,n_max

class RNN_BPTT():
    def __init__(self, topo=[2,16,1], n_iter=10000):
        self.topo = topo
        self.alpha = 0.1
        self.n_iter = n_iter
        self.pkl = 'data/rnn_bptt.pkl'
        try:
            with file(self.pkl,'rb') as f:
                self.wu = pickle.load(f)
                self.wv = pickle.load(f)
                self.ww = pickle.load(f)
        except:
            self.wu = 2 * np.random.random((topo[0], topo[1])) - 1
            self.wv = 2 * np.random.random((topo[1], topo[2])) - 1
            self.ww = 2 * np.random.random((topo[1], topo[1])) - 1


    def fit(self, d_bin, n):
        for idx in range(self.n_iter):
            a = np.random.randint(0,2 ** n) / 2
            b = np.random.randint(0,2 ** n) / 2
            c = a + b

            a_bin = d_bin[a]
            b_bin = d_bin[b]
            c_bin = d_bin[c]
            c_bin_calc = np.zeros_like(c_bin)
            # forward propagation
            g_err = 0
            lh = list()
            ldelta = list()
            lh.append(np.zeros((1,self.topo[1])))
            for i in range(n):
                X = np.array([[a_bin[-i - 1],b_bin[-i - 1]]])
                y = np.array([[c_bin[-i - 1]]])
                layer_h = simoid(np.dot(X, self.wu) + np.dot(lh[-1],self.ww))
                layer_o = simoid(np.dot(layer_h, self.wv))
                lh.append(copy.deepcopy(layer_h))
                ldelta.append((layer_o - y) * simoid(layer_o,True))
                g_err += ((layer_o - y)[0][0]) ** 2
                c_bin_calc[-i -1] = round(layer_o[0][0])

            # back propagation
            delta_v_update = np.zeros((self.topo[1], self.topo[2]))
            delta_w_update = np.zeros((self.topo[1], self.topo[1]))
            delta_u_update = np.zeros((self.topo[0], self.topo[1]))
            delta_h_next = np.zeros((1,self.topo[1]))
            for i in range(n):
                X = np.array([[a_bin[i],b_bin[i]]])
                delta_o = ldelta[-i - 1]
                delta_h = (np.dot(delta_o, self.wv.T) + np.dot(delta_h_next, self.ww.T)) * simoid(lh[-i -1],True)
                delta_v_update += np.dot(lh[-i -1].T, delta_o)
                delta_w_update += np.dot(lh[-i -2].T, delta_h)
                delta_u_update += np.dot(X.T, delta_h)
                delta_h_next = delta_h

            # update matrix of weight u,v w
            self.wu += -1 * self.alpha * delta_u_update
            self.wv += -1 * self.alpha * delta_v_update
            self.ww += -1 * self.alpha * delta_w_update
            # init parameters
            delta_u_update *= 0
            delta_v_update *= 0
            delta_w_update *= 0

            if idx % 1000 == 0:
                print "the total error is {0}".format(g_err)
                tmp = 0
                for idx,x in enumerate(reversed(c_bin_calc)):
                    tmp += x * pow(2, idx)
                print "training:{0} + {1} = {2}".format(a,b,tmp)
                print "----" * 10
                if tmp == a + b and g_err < 0.01:
                    break

        with open(self.pkl,'wb') as f:
            pickle.dump(self.wu,f)
            pickle.dump(self.wv,f)
            pickle.dump(self.ww,f)



    def predict(self, data, a, b):
        a_bin = data[a]
        b_bin = data[b]
        lh = list()
        ldelta = list()
        c_bin_calc = np.zeros_like(b_bin)
        c = 0
        lh.append(np.zeros((1,self.topo[1])))
        for i in range(n):
            X = np.array([[a_bin[-i - 1],b_bin[-i - 1]]])
            layer_h = simoid(np.dot(X, self.wu) + np.dot(lh[-1],self.ww))
            layer_o = simoid(np.dot(layer_h, self.wv))
            lh.append(copy.deepcopy(layer_h))
            c_bin_calc[-i -1] = round(layer_o[0][0])
        for idx,x in enumerate(reversed(c_bin_calc)):
            c += x * pow(2, idx)
        return c

if __name__ == "__main__":
    model = RNN_BPTT()
    data,n = load_data()
    model.fit(data,n)
    res = model.predict(data,113,15)
    print res