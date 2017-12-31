# coding:utf-8
__author__ = 'T'

"""
date: 2017-10-31
原理：通过贝叶斯公式将 P(Y|X) 转化为 求 P(X|Y) ，而P(Y)为先验概率，P(X|Y) 依据训练数据集很容得出。
备注：算法可能存在错误，没有考虑P(Y)；P(Y|X)=P(XY)/P(X)=P(Y)P(X|Y)/P(X)
"""

import csv
def loadCsv(filename):
    lines = csv.reader(open(filename,'rb'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

#dataset = loadCsv(filepath.rstrip(' ')+filename)
#print('load data file {0} with {1} rows').format(filename,len(dataset))

import random
def splitDataset(dataset,ratio):
    trainsize = int(len(dataset)*ratio)
    trainset = []
    copy = list(dataset)
    while len(trainset) < trainsize:
        index  = random.randrange(len(copy))
        trainset.append(copy.pop(index))
    return [trainset,copy]
"""
dataset = [[1],[2],[3],[4],[5]]
ratio = 0.67
train,test= splitDataset(dataset,ratio)
print 'split {0} rows into transet with {1} ,test with {2}'.format(len(dataset),train,test)
"""

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i];
        if  (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
"""
dataset = [[1,2,1],[0,1,0],[-1,3,1]]
se = separateByClass(dataset)
print 'separated instance {0}'.format(se)
"""
import math
def mean(nums):
    return sum(nums)/float(len(nums))

def stdev(nums):
    avg = mean(nums)
    variance = sum([pow(x-avg,2) for x in nums])/(len(nums)-1)
    return math.sqrt(variance)
"""
nums = [1,2,3,4,5]
print 'summary of  {0}:mean:{1},stdev:{2} '.format(nums,mean(nums),stdev(nums))
"""
def summarize(dataset):
    summaries = [(mean(attr),stdev(attr)) for attr in zip(*dataset)]
    del summaries[-1]
    return summaries

"""
dataset = [[1,20,1],[0,21,0],[-1,23,1]]
print zip(*dataset)
print 'attr summaries:{0}'.format(summarize(dataset))
"""

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classval,datasetval in separated.iteritems():
        summaries[classval] = summarize(datasetval)
    return summaries
"""
dataset = [[1,21,1],[2,12,0],[0,21,0],[-1,23,1]]
summaries = summarizeByClass(dataset)
print 'separated instance {0}'.format(summaries)
"""
# 生态分布概率计算
def calculateProbability(x,mu,sigma):
    expontent  = math.exp(-math.pow(x-mu,2)/(2*math.pow(sigma,2)))
    return  (1/(math.sqrt(2*math.pi)*sigma))*expontent

# 计算每个分类下发生概率
def calculateClssProbability(summaries,inputvector):
    probalities = {}
    for classval,classsummary in summaries.iteritems():
        probalities[classval] = 1
        for i in range(len(classsummary)):
            x = inputvector[i]
            mu,sigma = classsummary[i]
            probalities[classval]*=calculateProbability(x,mu,sigma)
    return probalities


# 取概率最大值，冒泡法
def predict(summaries,inputvector):
    summaries = calculateClssProbability(summaries,inputvector)
    bestlable,bestprob = None,-1
    for classval,classsummaries in summaries.iteritems():
        if classsummaries>bestprob:
            bestprob = classsummaries
            bestlable = classval
    return (bestlable,bestprob)

"""
summaris = {0:[(1,0.5)],1:[(20,5.0)]}
vec = [11,'']
result = predict(summaris,vec)
print 'the most possble class is:{0} and then probability is {1}'.format(result[0],result[1])
"""

def getPredictions(summaries,testset):
    predictions = []
    for i in range(len(testset)):
        result = predict(summaries,testset[i])
        predictions.append(result[0])
    return predictions

def getAccurancy(testset,predictions):
    currect = 0
    for i in range(len(testset)):
        if testset[i][-1] == predictions[i]:
            currect+=1
    return currect/float(len(testset))*100

def main():
    filepath = r'E:\PyData\dm\ '
    filename = 'pima-indians-diabetes.data.csv'
    ratio = 0.67
    dataset = loadCsv(filepath.rstrip(' ')+filename)
    trainingset,testset = splitDataset(dataset,ratio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingset), len(testset))
    summaries = summarizeByClass(trainingset)
    predictions = getPredictions(summaries,testset)
    accurancy = getAccurancy(testset,predictions)
    print 'Accurancy is {0}%'.format(accurancy)

main()