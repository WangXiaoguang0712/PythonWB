#coding:utf-8
__author__ = 'T'
import matplotlib.pyplot as plt

from twi.classify import dm_classify_dt as dt

decisionnode = dict(boxstyle='sawtooth',fc='0.8')
leafnode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')

#画线(末端带一个点)
def plotNode(nodetext,centerpt,parentpt,nodetype):
    plt.annotate(nodetext,xy = parentpt,xycoords='axes fraction',xytext=centerpt,textcoords='axes fraction', va="center", ha="center", bbox=nodetype, arrowprops=arrow_args)

#在指定位置添加文本
def plotMidText(cntrpt,parentpt,txtstring):
    xmid =  (parentpt[0]-cntrpt[0])/2.0+cntrpt[0]
    ymid =  (parentpt[1]-cntrpt[1])/2.0+cntrpt[1]
    plt.text(xmid,ymid,txtstring,va="center", ha="center", rotation=30)

#绘制决策树
def plotTree(mytree,parent_pt,nodetext):
    numleafs = dt.getNumLeafs(mytree)
    rootlable = mytree.keys()[0]

    # 定位第一棵子树的位置(这是蛋疼的一部分)
    cntr_pt = (plotTree.xOff + (1.0 + float(numleafs))/2.0/plotTree.totalW, plotTree.yOff)
    print plotTree.xOff,numleafs,plotTree.totalW, (1.0 + float(numleafs))/2.0/plotTree.totalW
    # 绘制当前节点到子树节点(含子树节点)的信息
    plotMidText(cntr_pt, parent_pt, nodetext)
    plotNode(rootlable, cntr_pt, parent_pt, decisionnode)

    substree = mytree[rootlable]
    # 开始绘制子树，纵坐标-1。
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in substree.keys():
        if type(substree[key]).__name__ == 'dict':
            # 子树分支则递归
            plotTree(substree[key],cntr_pt,str(key))
        else:
            # 叶子分支则直接绘制
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(substree[key], (plotTree.xOff, plotTree.yOff), cntr_pt, leafnode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntr_pt, str(key))
    # 子树绘制完毕，纵坐标+1。
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


#绘制有向线段(末端带一个节点)并显示
def createPlot(init_tree):
    plt.figure()
    plt.subplot(111)
    # 树的总宽度 高度
    plotTree.totalW = float(dt.getNumLeafs(init_tree))
    plotTree.totalD = float(dt.getTreeDepth(init_tree))
    # 当前绘制节点的坐标
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0

    # 绘制决策树
    plotTree(init_tree,(0.5,1.0),'111')
    plt.show()

def test():
    mydata,lables = dt.createDataSet()
    mytree = dt.createTree(mydata,lables)
    createPlot(mytree)

test()

