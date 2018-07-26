---
title: '机器学习:k-近邻算法'
date: 2018-07-26 19:10:10
tags: ['python','机器学习']
cayegories: 机器学习
toc: true
---



从本章开始我们开始机器学习的学习，在本章我们将探讨k-近邻算法的基本理论。

<!-- more -->

## 概述

k-近邻算法是一种基于分类与回归方法，在这里只讨论分类问题中的k-近邻算法。

三个基本要素：k值得选择、距离度量、分类决策规则

## 原理

1. 假设有一个带有标签的样本数据集，其中包含每条数据与所属分类的对应的关系

2. 输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较

   ​    a. 计算新数据与样本数据集中每条数据的距离

   ​    b. 对求得的所有距离进行排序

   ​    c. 取前k个样本数据对应的分类标签

3. 求k个数据中出现次数最多的分类标签作为新数据的分类



## 基本要素

1. k值得选择

   - 选择较小的k值容易发生过拟合，近似误差会减小，估计误差会增大
   - 选择较大的k值估计误差会减小，近似误差会增大

2. 距离度量

   在这里我们选择的是欧式距离公式

   假设有两个向量点A(x,y),B(x,y),则之间的距离为：

3. 分类决策规则

   入实例的 k 个邻近的训练实例中的多数类决定输入实例的类 

## k-近邻算法

```python
def classify0(inX,dataSet,labels,k):
    '''
    intX:用于分类的输入向量
    dataSet:输入的训练样本集
    labels:标签向量
    k:用于选择最近邻居的数目
    '''
    #1.距离计算
    dataSetSize=dataSet.shape[0]
    #生成和训练样本对应的矩阵，并与训练样本求差
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2#取平方
    #将矩阵每一行相加
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5#开方
    #距离排序,取其索引
    sortedDistIndicies=distances.argsort()

    #2.选择距离最小的k个点
    classCount={}
    for i in range(k):
        #找到该样本的类型
        voteIlabel=labels[sortedDistIndicies[i]]
        #在字典中将该类型加一
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #3.排序并返回出现最多的类型
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

```

