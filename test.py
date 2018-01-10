#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pickle


X = np.linspace(0,1,20)
y = [ -x * np.log(x) for x in X]

plt.figure()
plt.plot(X, y,)
plt.show()



