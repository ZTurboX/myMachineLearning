'''
决策树算法:ID3
步骤:
1.计算给定数据集的香农熵
2.划分数据集
3.构建决策树
4.使用决策树执行分类

ID3算法：
1.使用所有没有使用的属性并计算与之相关的样本熵值
2.选取其中熵值最小的属性
3.生成包含该属性的节点
'''


from math import log
import operator
import treePlotter

def calcShannonEnt(dataSet):
    '''
    计算给定数据集的香农熵
    '''

    numEntries=len(dataSet)

    #为所有可能分类创建字典
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    
    #以2为底数求对数
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries#类别出现的频率
        shannonEnt-=prob*log(prob,2)
    return shannonEnt


def splitDataSet(dataSet,axis,value):
    '''
    按照给定特征划分数据集
    dataSet:待划分的数据集
    axis:划分数据集的特征，表示每一行的axis列
    value:需要返回的特征的值，表示axis列对应的value值
    '''

    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    ID3,信息增益
    选择最好的数据集划分方式
    '''

    #求第一行有多少列的Feature
    numFeatures=len(dataSet[0])-1
    #数据集的原始信息熵
    baseEntropy=calcShannonEnt(dataSet)
    #最优信息增益值，最优Feature编号
    bestInfoGain,bestFeature=0.0,-1
    for i in range(numFeatures):
        #创建唯一的分类标签列表
        #获取对应的feature下的所有数据
        featList=[example[i] for example in dataSet]
        #获取去重后的集合
        uniqueVals=set(featList)
        #创建一个临时的信息熵
        newEntropy=0.0

        #遍历某一列的value集合，计算该列的信息熵
        #遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            #计算概率
            prob=len(subDataSet)/float(len(dataSet))
            #计算信息熵
            newEntropy+=prob*calcShannonEnt(subDataSet)
        
        #计算最好的信息增益
        #信息增益是熵的减少或数据无序度的减少
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature


def chooseBestFeatureToSplit2(dataSet):
    """
    c4.5,信息增益率
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy
        if (splitInfo == 0): # fix the overflow bug
            continue
        infoGainRatio = infoGain / splitInfo
        if (infoGainRatio > bestInfoGainRatio):
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''
    分类
    '''
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    '''
    创建树
    '''

    classList=[example[-1] for example in dataSet]
    
    #所有类标签完全相同，直接返回该类标签
    if classList.count(classList[0])==len(classList):
        return classList[0] 
    #使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    #选择最优的列，得到最优列对应的label含义
    bestFeat=chooseBestFeatureToSplit(dataSet)
    #获取label的名称
    bestFeatLabel=labels[bestFeat]
    #初始化myTree
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    #取出最优列，对branch做分类
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        #求剩余的标签label
        subLabels=labels[:]
        #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用createTree()
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


def classify(inputTree,featLabels,testVec):
    '''
    测试算法，使用决策树执行分类
    inputTree:决策树模型
    featLabels:Feature标签对应的名称
    testVec:测试输入的数据
    '''
    #获取tree的根节点对应的key值
    firstStr=list(inputTree.keys())[0]
    #通过key得到根节点对应的value
    secondDict=inputTree[firstStr]
    #判断根节点名称获取根节点在label中的先后顺序，判断输入的testVec怎样开始对照数来做分类
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel



def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels


'''
使用pickle模块存储决策树
'''
def storeTree(inputTree,fileName):
    import pickle
    fw=open(fileName,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)


myDat,labels=createDataSet()
myTree=treePlotter.retrieveTree(0)
storeTree(myTree,'classifierStorage.txt')
grabTree('classifierStorage.txt')
