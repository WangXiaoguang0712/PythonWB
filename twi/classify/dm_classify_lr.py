#coding:utf-8
__author__ = 'T'

import math
import numpy as np
import  pandas as pd
from matplotlib import pyplot as plt
from  sklearn import  datasets
import  time

class logisticRegression(object):
    theta = []
    complexity = 0
    thetaline = [[]]
    def loadData(self):
        train_x = []
        train_y = []
        file = r'E:\PyData\dm\lr_data.txt'
        f = open(file,'r')
        for line in f.readlines():
            linearr = line.strip().split()
            train_x.append([1.0,float(linearr[0]),float(linearr[1])])
            train_y.append(float(linearr[2]))
        return np.matrix(train_x),np.matrix(train_y).T

    def simoid(self,x):
        return  1.0/(1+math.exp(-x))

    def getResult(self,args):
        val = lambda x: 1 if x>=0.5 else 0
        return val(self.simoid(args))

    def fit(self,X,Y,opts):
        if opts['optimizeType'] == 'GD':
            self.theta = self.gradAscent(X,Y)
        elif opts['optimizeType'] == 'SGD':
            self.theta = self.stochasticGradeAscent(X,Y,opts)
        elif opts['optimizeType'] == 'OSGD':
            self.theta = self.optimizeStochasticGradeAscent(X,Y,opts)
        else:
            raise NameError('not support optimize type')
        return  self.theta


        #return output
    def gradAscent(self,X,Y):
        samples,features = X.shape
        weights = np.ones((features,1))
        alpha = opts['alpha']
        cycles = opts['cycles']
        epsilon = opts['epsilon']
        error = np.zeros(features)
        for i in range(cycles):
            val = np.array(map(self.simoid,[X[i].dot(weights) for i in range(samples)])).reshape(samples,1)
            diff = Y - val
            weights = weights + alpha*(X.T.dot(diff))
            self.complexity += samples
            if np.linalg.norm(weights-error) < epsilon:
                break
            else:
                error = weights
        return weights

    def stochasticGradeAscent(self,X,Y,opts):
        samples,features = X.shape
        weights = np.zeros((features,1))
        self.thetaline = weights.reshape(1,-1)
        alpha = opts['alpha']
        cycles = opts['cycles']
        epsilon = opts['epsilon']
        error = np.zeros(features)
        for i in range(cycles):
            for j in range(samples):
                val =  self.simoid(X[j].dot(weights))
                diff = Y[j] - val
                weights = weights + alpha*(X[j].T.dot(diff))
                self.thetaline = np.concatenate((self.thetaline,weights.reshape(1,-1)),axis=0)
                self.complexity += 1
            if np.linalg.norm(weights-error) < epsilon:
                break
            else:
                error = weights
        return weights

    def optimizeStochasticGradeAscent(self,X,Y,opts):
        samples,features = X.shape
        weights = np.ones((features,1))
        self.thetaline = weights.reshape(1,-1)
        alpha = opts['alpha']
        cycles = opts['cycles']
        epsilon = opts['epsilon']
        error = np.zeros(features)
        for i in range(cycles):
            for j in range(samples):
                alpha = 0.4 / (1.0 + j*i) + 0.001
                val =  self.simoid(X[j].dot(weights))
                diff = Y[j] - val
                weights = weights + alpha*(X[j].T.dot(diff))
                self.thetaline = np.concatenate((self.thetaline,weights.reshape(1,-1)),axis=0)
                self.complexity += 1
            if np.linalg.norm(weights-error) < epsilon:
                break
            else:
                error = weights
        return weights

    def testAccurency(self,X,Y):
        samples,features = X.shape
        output = X.dot(self.theta)
        val = map(self.getResult,[output[i] for i in range(samples)])
        mat_rs = Y == np.matrix(val).T
        return float(np.sum(mat_rs))/samples

    def showLR(self,X,Y):
        samples,features = X.shape
        if features != 3:
            print 'sorry!'
            return 1
        plt.figure()
        plt.subplot(111)
        for i in range(samples):
            if int(Y[i,0]) == 0:
                plt.plot(X[i,1],X[i,2],'or')
            elif int(Y[i,0]) == 1:
                plt.plot(X[i,1],X[i,2],'+b')
        min_x = min(X[:,1])[0,0]
        max_x = max(X[:,1])[0,0]
        weights = self.theta.getA()
        y_min_x = float(-weights[0]-weights[1]*min_x)/weights[2]
        y_max_x = float(-weights[0]-weights[1]*max_x)/weights[2]
        plt.plot([min_x,max_x],[y_min_x,y_max_x],'-g')
        plt.xlabel('X1');plt.ylabel('X2')
        plt.show()

    def  showTheta(self):
        plt.figure()
        plt.subplot(311)
        plt.plot([x for x in range(self.complexity)],[self.thetaline[i,0] for i in range(self.complexity)],'-r')
        plt.xlabel('cycle');plt.ylabel('theta0')
        plt.subplot(312)
        plt.plot([x for x in range(self.complexity)],[self.thetaline[i,1] for i in range(self.complexity)],'-g')
        plt.xlabel('cycle');plt.ylabel('theta1')
        plt.subplot(313)
        plt.plot([x for x in range(self.complexity)],[self.thetaline[i,2] for i in range(self.complexity)],'-b')
        plt.xlabel('cycle');plt.ylabel('theta2')
        plt.show()

if __name__ == "__main__":
    lr = logisticRegression()
    X,Y = lr.loadData()
    test_x,test_y = X,Y

    opts = {'optimizeType':'GD','alpha':0.001,'cycles':1000,'epsilon':1e-3}
    lr.fit(X,Y,opts)
    print lr.thetaline[:10]
    print lr.complexity
    #rs = lr.testAccurency(test_x,test_y)
    lr.showLR(X,Y)
    lr.showTheta()
