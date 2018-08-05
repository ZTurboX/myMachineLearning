# 使用Apriori算法进行关联分析

## 关联分析

关联分析是一种在大规模数据集中寻找有趣关系的任务。这些关系可以有两种形式：频繁项集或者关联规则。频繁项集是经常出现在一块的物品的集合，关联规则暗示两种物品之间可能存在很强的关系。

一个项集的支持度被定义为数据集中包含该项集的记录所占的比例。

可信度: 针对一条诸如 {尿布} -> {葡萄酒} 这样具体的关联规则来定义的。这条规则的 `可信度` 被定义为 `支持度({尿布, 葡萄酒})/支持度({尿布})` 

## Apriori算法

### 原理

Apriori 算法是发现 频繁项集的一种方法。 Apriori 算法的两个输入参数分别是最小支持度和数据集。 该算法首先会生成所有单个物品的项集列表。 接着扫描交易记录来查看哪些项集满足最小支持度要求，那些不满足最小支持度要求的集合会被去掉。 燃尽后对生下来的集合进行组合以声场包含两个元素的项集。 接下来再重新扫描交易记录，去掉不满足最小支持度的项集。 该过程重复进行直到所有项集被去掉。 

### 生成候选项集

对数据集中的每条交易记录tran

对每个候选项集can:

&nbsp; &nbsp; &nbsp; &nbsp; 检查can是否是tran的子集：

&nbsp; &nbsp; &nbsp; &nbsp; 如果是，增加can的计数值

对每个候选项集：

&nbsp; &nbsp; &nbsp; &nbsp; 如果支持度不低于最小值，则保留该项集&nbsp; 

&nbsp; &nbsp; &nbsp; &nbsp; 返回所有频繁项集列表

```python
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    '''
    创建一个大小为1的候选项集的集合
    '''
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #frozenset为用户不能改变数据的类型
    return list(map(frozenset,C1))

def scanD(D,Ck,minSupport):
    '''
    计算候选数据集Ck在数据集D中的支持度，并返回支持度大于最下支持度的数据
    Ck:候选项集列表
    minSupport:最下支持度
    '''
    ssCnt={}
    for tid in D:
        for can in Ck:
            #检查是否can中的每个元素都在tid中
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
    numItems=float(len(D))
    retList=[]
    supportData={}
    for key in ssCnt:
        support=ssCnt[key]/numItems
        if support>=minSupport:
            retList.insert(0,key)
        supportData[key]=support
    return retList,supportData
```

### Apriori算法

当集合中项的个数大于0时：

&nbsp; &nbsp; &nbsp; &nbsp; 构建一个k个项组成的候选项集的列表

&nbsp; &nbsp; &nbsp; &nbsp; 检查数据以确认每个项集都是频繁的

&nbsp; &nbsp; &nbsp; &nbsp; 保留频繁项集并构建k+1项组成的候选项集的列表

```python
def aprioriGen(Lk,k):
    '''
    输入频繁项集LK与项集元素个数k，输出候选项集Ck
    '''
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2]
            L2=list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            #前k-2个项相同时，将两个集合合并
            if L1==L2:
                #|为集合合并操作
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    '''
    Apriori算法：
    '''
    C1=createC1(dataSet)
    D=list(map(set,dataSet))
    L1,supportData=scanD(D,C1,minSupport)
    L=[L1]
    k=2
    while len(L[k-2])>0:
        Ck=aprioriGen(L[k-2],k)
        Lk,supK=scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData
```

### 从频繁项集中挖掘关联规则

```python
def calcConf(freqSet,H,supportData,brl,minConf=0.7):
    '''
    计算可信度
    freqSet:频繁项集的元素,如[1,3]
    H:频繁项集中的元素的集合,如[1,3]中的[1],[3]
    brl:关联规则列表
    minConf:最小可信度
    '''
    #记录可信度大于最小可信度的集合
    prunedH=[]
    for conseq in H:
        conf=supportData[freqSet]/supportData[freqSet-conseq]
        if conf>=minConf:
            print(freqSet-conseq,"-->",conseq,"conf:",conf)
            brl.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,brl,minConf=0.7):
    '''
    递归计算频繁项集的规则
    '''
    m=len(H[0])
    if len(freqSet)>(m+1):
        Hmp1=aprioriGen(H,m+1)
        Hmp1=calcConf(freqSet,Hmp1,supportData,brl,minConf)
        if len(Hmp1)>1:
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)

def generateRules(L,supportData,minConf=0.7):
    '''
    生成关联规则
    '''
    #可信度规则列表
    bigRuleList=[]
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1=[frozenset([item]) for item in freqSet]
            if i>1:
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList
```

