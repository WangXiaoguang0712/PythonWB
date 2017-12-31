#coding:utf-8
__author__ = 'T'
import numpy as np
import math
import copy

#========================
#计算香浓熵
#输入：数据集
#输出：熵
#=========================
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    classCount = {}
    for vector in dataSet:
        if vector[-1] not in classCount:
            classCount[vector[-1]]=0
        classCount[vector[-1]] += 1
    #计算熵
    Entropy = 0.0
    for key,val in classCount.iteritems():
        Probability = float(val)/numEntries
        Entropy+= -Probability*math.log(Probability,2)
    return Entropy

#创建测试数据集
def createDataSet():
    dataSet = [[1,0,1,1,'yes'],
            [0,1,1,1,'yes'],
            [1,0,1,1,'no'],
            [1,0,0,1,'no'],
            [0,0,0,1,'no']]
    labels = ['is rich','is handsom','is height','issteady']
    return [dataSet,labels]

#划分数据集::可优化
def splitDataSet(dataset,axis,val):
    retSet = []
    for vec in dataset:
        if vec[axis] == val:
            veccp = copy.copy(vec)
            veccp.pop(axis)
            retSet.append(veccp)
    return [len(retSet),retSet]

#选择最好的特征进行划分
def chooseBestFeatureToSplit(dataset):
    bestfeature = -1 #最好特征
    bestinfogain = 0 #最大熵增量
    baseentropy = calcShannonEnt(dataset)
    numofdataset = len(dataset)#数据集大小
    numoffeature = len(dataset[0]) -1 #特征数

    for i in range(numoffeature):
        newentropy = 0.0
        valueoffeatue = [item[i] for item in dataset]
        uniquevalueoffeature = set(valueoffeatue)
        for uitem in uniquevalueoffeature:
            numsset,retset = splitDataSet(dataset,i,uitem)
            proofitem = float(numsset)/numofdataset
            ss = calcShannonEnt(retset)
            newentropy += proofitem * ss
        infogain = baseentropy - newentropy
        if infogain >= bestinfogain:
            bestinfogain = infogain
            bestfeature = i
    return bestfeature

# 出现次数最多的分类
def getMaxClass(list):
    classcount = {}
    for item in list:
        if item not in classcount:
            classcount[item] = 0
        classcount[item] += 1
    sortedclasslabel = sorted(classcount.iteritems(),key=lambda x:x[1],reverse=True)
    return sortedclasslabel[0][0]

def createTree(dataset,labels):
    classlist = list(set([x[-1] for x in dataset]))

    if len(classlist) == 1:
        return classlist[0]
    if len(dataset[0]) == 1:
        return getMaxClass(dataset[0])

    bestfeature = chooseBestFeatureToSplit(dataset)
    bestfeaturelabel = labels[bestfeature]
    mytree = {bestfeaturelabel:{}}
    del(labels[bestfeature])

    bestfeatureval = [x[bestfeature] for x in dataset]
    ubestfeatureval = set(bestfeatureval)
    for item in ubestfeatureval:
        sublabel = labels[:]
        mytree[bestfeaturelabel][item] = createTree(splitDataSet(dataset,bestfeature,item)[1],sublabel)
    return mytree

#获取叶节点数
def getNumLeafs(mytree):
    numleafs = 0
    root = mytree.keys()[0]
    subtree = mytree[root]
    for key in subtree.keys():
        if type(subtree[key]).__name__ == 'dict':
            numleafs += getNumLeafs(subtree[key])
        else:
            numleafs += 1
    return numleafs

#获取树的深度
def getTreeDepth(mytree):
    branchdepth = 0
    maxdepth = 1
    rootlable = mytree.keys()[0]
    subtree = mytree[rootlable]
    for item in subtree.keys():
        if type(subtree[item]).__name__ == 'dict':
            branchdepth = 1 + getTreeDepth(subtree[item])
        else:
            branchdepth = 1
        if branchdepth > maxdepth:
            maxdepth = branchdepth
    return maxdepth

#存储决策树
def storeTree(input_tree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(input_tree,fw)
    fw.close()

#提取决策树
def grabTree(filename):
    import pickle
    fr = open(file)
    return  pickle.load(fr)

#分类
def classify(input_tree,labels,vector):
    dest_class = ''
    rootlable = input_tree.keys()[0]
    subtree = input_tree[rootlable]
    #获取索引
    index = labels.index(rootlable)
    #获取特征值
    feature_val = vector[index]
    #获取最终结果
    try:
        tree_val = subtree[feature_val]
    except:
        return "i don't know"
    if isinstance(tree_val,dict):
        dest_class = classify(tree_val,labels,vector)
    else:
        dest_class = tree_val
    return dest_class


def main():
    mydata,lables = createDataSet()
    mytree = createTree(mydata,lables)

    myDat, labels = createDataSet()
    print mytree
    print classify(mytree, labels, [0,0,1,1])

main()

