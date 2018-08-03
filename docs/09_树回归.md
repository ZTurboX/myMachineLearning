# 树回归

## CART算法

### 原理

第三章的ID3算法不能处理连续型特征，但使用CART算法能用二元切分法处理连续型变量“如果特征值大于给定值就走左子树，否则就走右子树。

### 算法

对每个特征：

&nbsp; &nbsp; &nbsp; 对每个特征值：

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 将数据集切分成两份

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 计算切分的误差

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 如果当前误差小于当前最小误差，那么就将当前切分设定为最佳切分并更新最小误差

返回最佳切分的特征和阈值

```python
def binSplitDataSet(dataSet,feature,value):
    '''
    在给定特征和特征值的情况下，通过数组过滤方式将上述数据集切分得到两个子集
    dataSet:数据集
    feature:待切分的特征列
    value:特征列要比较的值
    '''
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1


def regLeaf(dataSet):
    '''
    生成叶节点
    '''
    return mean(dataSet[:,-1])

def regErr(dataSet):
    '''
    在给定数据上计算目标变量的平方误差
    '''
    return var(dataSet[:,-1])*shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''
    用最佳切分方式切分数据集，并且生成相应的叶节点
    dataSet:原始数据集
    leafType:建立叶子结点
    errType:误差计算
    ops:[容许误差下降值，切分的最小样本数]
    '''
    #最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分
    tolS=ops[0]
    #划分最小样本数小于，就不继续划分
    tolN=ops[1]
    
    #取矩阵的最后一列，并转换为数组取第0列，如果size为1，不用继续划分
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)
    m,n=shape(dataSet)
    #无分类误差的总平方和
    S=errType(dataSet)
    bestS,bestIndex,bestValue=inf,0,0
    #循环处理每一列对应的feature值
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            #对该列进行分组，然后组内的成员的val值进行二元划分
            mat0,mat1=binSplitDataSet(dataSet,featIndex,splitVal)
            if shape(mat0)[0]<tolN or shape(mat1)[0]<tolN:
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    #如果误差减少不大则退出
    if (S-bestS)<tolS:
        return None,leafType(dataSet)
    mat0,mat1=binSplitDataSet(dataSet,bestIndex,bestValue)
    if shape(mat0)[0]<tolN or shape(mat1)[0]<tolN:
        return None,leafType(dataSet)
    return bestIndex,bestValue

'''
构件树：
找到最佳的待切分特征：
    如果该节点不能再分，将该节点存为叶节点
    执行二元切分
    在右子树调用createTree()
    在左子树调用createTree()
'''
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''
    获取回归树
    '''
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat is None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree
```

## 预剪枝

顾名思义，预剪枝就是及早的停止树增长，在构造决策树的同时进行剪枝。

所有决策树的构建方法，都是在无法进一步降低熵的情况下才会停止创建分支的过程，为了避免过拟合，可以设定一个阈值，熵减小的数量小于这个阈值，即使还可以继续降低熵，也停止继续创建分支。但是这种方法实际中的效果并不好。

## 后剪枝

基于已有的树切分测试数据：

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 如果存在任一子集是一棵树，则在该子集递归剪枝过程

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 计算将当前两个叶节点合并后的误差

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 计算不合并的误差

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 如果合并会降低误差的话，就将叶节点合并

```python
def isTree(obj):
    '''
    判断树是否是字典
    '''
    return (type(obj).__name__=='dict')

def getMean(tree):
    '''
    从上到下遍历树直到叶节点，如果找到两个叶节点则计算均值
    '''
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def proun(tree,testData):
    '''
    从上到下找到叶节点，用测试数据集判断这些叶节点合并是否能降低测试误差
    '''
    if shape(testData)[0]==0:
        return getMean(tree)
    #如果树是字典，则切分
    if isTree(tree['right']) or isTree(tree['left']):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    #如果左边分支是字典，就传入左边的数据集和左边的分支
    if isTree(tree['left']):
        tree['left']=proun(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right']=proun(tree['right'],rSet)
    #如果左右两边是叶子结点，分割测试数据集
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        #不合并的总方差
        errorNoMerge=sum(power(lSet[:,-1] - tree['left'],2))+sum(power(rSet[:,-1] - tree['right'],2))
        #合并的总方差
        treeMean=(tree['left']+tree['right'])/2.0
        errorMerge=sum(power(testData[:,-1]-treeMean,2))
        if errorMerge<errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree
```

## 模型树

模型树就是把叶节点设定为分段线性函数，分段线性是指模型由多个线性片段组成

```python
'''
模型树
'''
def linearSolve(dataSet):
    '''
    将数据集格式化成目标变量Y和自变量X，执行标准线性回归，得到ws
    '''
    m,n=shape(dataSet)
    X=mat(ones((m,n)))
    Y=mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]
    Y=dataSet[:,-1]
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise NameError('this matrix is singular, cannot not inverse.\ntry increasing the second value of ops')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    '''
    当数据集不需要切分的时候，生成叶节点的模型
    '''
    ws,X,Y=linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    '''
    计算误差,返回yHat和Y之间的平方误差
    '''
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))


def regTreeEval(model,inDat):
    '''
    回归树测试案例
    model:指定模型
    inDat:输入测试数据
    '''
    return float(model)

def modelTreeEval(model,inDat):
    '''
    模型树测试案例
    '''
    n=shape(inDat)[1]
    X=mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    '''
    给定模型的树进行预测。自顶向下遍历树，直到叶节点
    '''
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']]<=tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    '''
    预测结果
    '''
    m=len(testData)
    yHat=mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat
```

