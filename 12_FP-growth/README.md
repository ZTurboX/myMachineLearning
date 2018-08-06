# FP-growth算法

## FP-growth原理

基于数据构建FP树

1. 遍历所有的数据集合，计算所有项的支持度
2. 丢弃非频繁的项
3. 基于支持度降序排序所有的项

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/12_FP-growth/1.PNG)

4. 所有数据集合按照得到的顺序重新整理。
5. 重新整理完成后，丢弃每个集合末尾非频繁的项。 

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/12_FP-growth/2.PNG)

6.  读取每个集合插入FP树中，同时用一个头部链表数据结构维护不同集合的相同项。 

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/12_FP-growth/3.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/12_FP-growth/4.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/12_FP-growth/5.PNG)

得到的FP树

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/12_FP-growth/6.PNG)

```python
def createTree(dataSet,minSup=1):
    '''
    创建Fp-tree:
    1.遍历数据集获得每个元素项的出现频率
    2.去掉不满足最小支持度的元素项
    3.构建FP树：
        读入每个项集并将其添加到一条已存在的路径中：
            如果该路径不存在，则创建一条新路径

    minSup:最小的支持度
    '''
    #支持度>=minSup的字典
    headerTable={}
    for trans in dataSet:
        #统计每一行中每个元素出现的总次数
        for item in trans:
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]
    #删除headerTable中，元素次数<最小支持度的元素
    for k in list(headerTable.keys()):
        if headerTable[k]<minSup:
            del(headerTable[k])
    #满足minSup元素集合
    freqItemSet=set(headerTable.keys())
    if len(freqItemSet)==0:
        return None,None
    for k in headerTable:
        #格式化:dist{key:[元素次数,None]}
        headerTable[k]=[headerTable[k],None]
    #创建树
    retTree=treeNode('Null Set',1,None)
    for tranSet,count in dataSet.items():
        localD={}
        #根据频率对每个事物中的元素排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item]=headerTable[item][0]
        if len(localD)>0:
            #取出元组的key值
            orderedItems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree,headerTable

```

## 从FP树中挖掘频繁项集

步骤：

1. 从FP树中获得条件模式基
2. 利用条件模式基，构建一个条件Fp树
3. 迭代重复步骤1和2，直到树包含一个元素为止

条件模式基是以所查找元素项为结尾的路径集合。每一条路径都是一条前缀路径。

```python
def ascendTree(leafNode,prefixPath):
    '''
    如果父节点存在，就记录当前节点的name值

    leafNode:要查询的节点所在当前nodeTree
    prefixPath:前缀路径
    '''
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

def findPrefixPath(basePat,treeNode):
    '''
    basePat:要查询的节点值
    treeNode:查询的节点所在的当前nodeTree
    '''
    #对非basePat的倒序值作为key,赋值为count
    conPats={}
    while treeNode is not None:
        prefixPath=[]
        #寻找该节点的父节点
        ascendTree(treeNode,prefixPath)
        if len(prefixPath)>1:
            conPats[frozenset(prefixPath[1:])]=treeNode.count
        treeNode=treeNode.nodeLink
    return conPats

def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    '''
    创建条件FP树
    headerTable:满足minSup所有的元素字典
    minSup:最小支持项集
    freqItemList:存储频繁子项的列表
    '''
    #从小到大排序，得到频繁项集的key
    bigL=[v[0] for v in sorted(headerTable.items(),key=lambda p:p[1][0])]
    for basePat in bigL:
        newFreqSet=preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases=findPrefixPath(basePat,headerTable[basePat][1])
        myCondTree,myHead=createTree(condPattBases,minSup)
        if myHead is not None:
            myCondTree.disp(1)
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)

```

