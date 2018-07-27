# 决策树

## 概述

决策树(Decision Tree)算法是一种分类和回归算法，典型算法有ID3、C4.5、CART，本章我们使用的是ID3算法。

构建决策树算法的步骤：特征选择、决策树的生成、决策树的修剪

## 定义

在机器学习中，决策树是一个预测模型。它代表的是对象属性与对象值之间的一种映射关系。树中每个节点表示某个对象，而每个分叉路径代表某个可能的属性值，每个叶节点对应从根节点到该叶节点所经历的路径表示的是对象的值。

一个邮件分类系统的例子：

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/03_DecisionTree/%E9%82%AE%E4%BB%B6%E5%88%86%E7%B1%BB%E7%B3%BB%E7%BB%9F.PNG)

## 基本原理

### 信息增益

集合信息的度量方式称为**香农熵**或**熵**

![信息熵](https://github.com/TonyJent/myMachineLearning/blob/master/images/03_DecisionTree/%E4%BF%A1%E6%81%AF%E7%86%B5.PNG)

在划分数据集之前之后信息发生的变化成为**信息增益**

![信息增益](https://github.com/TonyJent/myMachineLearning/blob/master/images/03_DecisionTree/%E4%BF%A1%E6%81%AF%E5%A2%9E%E7%9B%8A.PNG)

### ID3算法(迭代二叉树3代)

1. 使用所有没有使用的属性并计算与之相关的样本的熵值
2. 选取其中熵值最小的属性
3. 生成包含该属性的节点

## 核心代码

第一步：计算给定数据集的香农熵

```python
from math import log
import operator

def calcShannonEnt(dataSet):23
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
```

第二步：按照给定特征划分数据集

```python
def chooseBestFeatureToSplit(dataSet):
    '''
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

```

第三步：构建树

```python
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
```

第四步：使用决策树分类

```python
def classify(inputTree,featLabels,testVec):
    '''
    测试算法，使用决策树执行分类
    inputTree:决策树模型
    featLabels:Feature标签对应的名称
    testVec:测试输入的数据
    '''
    #获取tree的根节点对应的key值
    firstStr=inputTree.keys()[0]
    #通过key得到根节点对应的value
    secondDict=inputTree[firstStr]
    #判断根节点名称获取根节点在label中的先后顺序，判断输入的testVec怎样开始对照数来做分类
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDictp[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel
```

## 项目实战

[使用Matplotlib注解绘制树形图](https://github.com/TonyJent/myMachineLearning/blob/master/03_DecisionTree/treePlotter.py)

[使用决策树预测隐形眼镜类型](https://github.com/TonyJent/myMachineLearning/blob/master/03_DecisionTree/lensesTest.py)

