import numpy as np
from numpy import logspace

def getelnalpha(A,B,PI,O,elnalpha):
    N,K = np.shape(B)
    T = np.shape(O)[0]
    for i in range(N):
        elnalpha[0][i] = logSpace.elnproduct(logSpace.eln(PI[i]),logSpace.eln(B[i][O[0]]))
    for t in range(1,T):
        for j in range(N):
            logalpha = logSpace.LOGZERO
            for i in range(N):
                logalpha = logSpace.elnsum(logalpha,logSpace.elnproduct(elnalpha[t-1][i],logSpace.eln(A[i][j])))
            elnalpha[t][j] = logSpace.elnproduct(logalpha,logSpace.eln(B[j][O[t]]))
    return True
def getelnbeta(A,B,PI,O,elnbeta):
    N,K = np.shape(B)
    T = np.shape(O)[0]
    for i in range(N):
        elnbeta[T-1][i] =0
    for t in range(T-2,-1,-1):
        for i in range(N):
            logbeta = logSpace.LOGZERO
            for j in range(N):
                logbeta = logSpace.elnsum(logbeta,logSpace.elnproduct(logSpace.eln(A[i][j]),logSpace.elnproduct(logSpace.eln(B[j][O[t+1]]),elnbeta[t+1][j])))
            elnbeta[t][i] = logbeta
    return True
def getelngamma(elnalpha,elnbeta,elngamma):
    T,N = np.shape(elngamma)

    for t in range(T):
        normalizer = logSpace.LOGZERO
        for i in range(N):
            elngamma[t][i] = logSpace.elnproduct(elnalpha[t][i],elnbeta[t][i])
            normalizer = logSpace.elnsum(normalizer,elngamma[t][i])
        for i in range(N):
            elngamma[t][i] = logSpace.elnproduct(elngamma[t][i],-1.0*normalizer)
    return True
def getelnxi(elnalpha,elnbeta,elnxi,A,B,O):
    T_1,N = np.shape(elnxi)[:2]
    T = T_1+1

    for t in range(T-1):
        normalizer = logSpace.LOGZERO
        for i in range(N):
            for j in range(N):
                Bbeta = logSpace.elnproduct(logSpace.eln(B[j][O[t+1]]),elnbeta[t+1][j])
                elnxi[t,i,j] = logSpace.elnproduct(elnalpha[t][i],logSpace.elnproduct(logSpace.eln(A[i][j]),Bbeta))
                normalizer = logSpace.elnsum(normalizer,elnxi[t,i,j])
        for i in range(N):
            for j in range(N):
                elnxi[t,i,j] = logSpace.elnproduct(elnxi[t,i,j],-normalizer)

    return True
def getpi(elngamma,nPI):
    N= np.shape(nPI)[0]

    for i in range(N):
        nPI[i] = logSpace.eexp(elngamma[0][i])
    return True
def geta(elngamma,elnxi,nA):
    T_1,N = np.shape(elnxi)[:2]
    T = T_1+1

    for i in range(N):
        normalizer = logSpace.LOGZERO
        numerators = [logSpace.LOGZERO for j in range(N)]
        for t in range(T-1):
            normalizer = logSpace.elnsum(normalizer,elngamma[t][i]) 
            for j in range(N):
                numerators[j] = logSpace.elnsum(numerators[j],elnxi[t,i,j])
        for j in range(N):            
              nA[i][j] = logSpace.eexp(logSpace.elnproduct(numerators[j],-normalizer))

    return True
def getb(elngamma,elnxi,O,nB):

    T = np.shape(O)[0]
    N,K = np.shape(nB)

    for j in range(N):
        for k in range(K):
            numerator = logSpace.LOGZERO
            normalizer = logSpace.LOGZERO
            for t in range(T):
                if O[t] == k:
                    numerator = logSpace.elnsum(numerator,elngamma[t][j])
                normalizer = logSpace.elnsum(normalizer,elngamma[t][j])
            nB[j][k] = logSpace.eexp(logSpace.elnproduct(numerator,-1.0*normalizer))
    return nB
def BaumWelch(O,A,B,PI,criterion=0.001):
    # N,K,T确定了A,B和PI的size
    T = np.shape(O)[0]
    N,K = np.shape(B)

    #参数初始化
    #A = np.random.rand(N,N)
    #A = np.divide(A,A.sum(axis=1).reshape(N,1))
    #B = np.random.rand(N,K)
    #B = np.divide(B,B.sum(axis=1).reshape(N,1))
    #PI = np.random.rand(N)
    #PI /= np.sum(PI)


    #为了提高速度，在迭代之前就为迭代过程中的中间变量开辟内存空间，迭代过程中不在开辟新空间
    elnbeta = np.zeros((T,N)) 
    elnalpha = np.zeros((T,N))
    elngamma = np.zeros((T,N))
    elnxi = np.zeros((T-1,N,N))

    nA = np.zeros((N,N))
    nB = np.zeros((N,K))
    nPI = np.zeros(N)

    done = False
    iters = 0
    while not done:
        iters += 1
        #print iters
        getelnalpha(A,B,PI,O,elnalpha)
        getelnbeta(A,B,PI,O,elnbeta)
        getelngamma(elnalpha,elnbeta,elngamma)
        getelnxi(elnalpha,elnbeta,elnxi,A,B,O)

        getpi(elngamma,nPI)
        geta(elngamma,elnxi,nA)
        getb(elngamma,elnxi,O,nB)

        if np.max(np.abs(nPI-PI))<= criterion and np.max(np.abs(nA-A))<= criterion and np.max(np.abs(nB-B))<= criterion:
            done = True #这种情况下视为收敛
        else:# 将更新后的参数nPI,nA,nB备份到PI,A,B
            np.copyto(PI,nPI)
            np.copyto(A,nA)
            np.copyto(B,nB)

    return A,B,PI