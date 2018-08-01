# 集成学习(Ensemble Learning)

## 概述

集成学习(Ensemble Learning)是通过构建并结合多个学习器来完成学习任务。![](https://github.com/TonyJent/myMachineLearning/blob/master/images/07_Ensemble%20Learning/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0.PNG)

集成学习方法可以分为两类:

1. 个体学习器间存在强依赖关系，必须串行生成的序列化方法，代表是Boosting
2. 个体学习器间不存在强依赖关系，可同时生成的并行化方法，代表是Bagging和随机森林

## AdaBoost

AdaBoost运行过程：训练数据中的每个样本，并赋予一个权重，这些权重构成向量D。一开始，这些权重都初始化成相等值。首先在训练数据上训练出一个弱分类器并计算该分类器的错误率，然后在同一数据集上再次训练弱分类器。在分类器的第二次训练中，将会重新调整每个样本的权重，其中第一次分对的样本的权重将会降低，而第一次分错的样本的权重将会提高。

### 相关公式

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/07_Ensemble%20Learning/AdaBoost%E5%85%AC%E5%BC%8F.PNG)

## 随机森林

随机森林是利用多棵树对样本进行训练兵预测的一种分类器。

随机森林构建有两个方面：数据的随机性和待选特征的随机化，使得随机森林中的决策树都能彼此不同，提升系统的多样性，从而提升分类性能。

## AdaBoost核心算法

1. 通过阈值比较对数据进行分类：所有在阈值一边的数据分为类别-1,另一边的分为+1 

   ```python
   def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
       '''
       通过阈值比较对数据进行分类：所有在阈值一边的数据分为类别-1,另一边的分为+1
       dataMatrix:数据集
       dimen:特征的哪一列
       threshVal:特征要比较的值
       '''
       #返回数据全置为1
       retArray=ones((shape(dataMatrix)[0],1))
       if threshIneq=='lt':
           #要修改左边的值
           retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
       else:
           #修改右边的值 
           retArray[dataMatrix[:,dimen]>threshVal]=-1.0
       return retArray
   ```

   

2. 找到数据集上最佳的单层决策树

    单层决策树算法：

   将最小错误率minError设为正无穷大

   &nbsp;将最小错误率minError设为正无穷大

   对数据集中的每个特征：

   &nbsp; &nbsp; &nbsp; &nbsp; 对每个步长：&nbsp; 

   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 对每个不等号:

   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 建立一棵单层决策树并利用加权数据集对它测试

   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 如果错误率低于minError,则将当前单层决策树设为最佳单层决策树

   返回最佳单层决策树

```python
def buildStump(dataArr,classLabels,D):
    '''
    找到数据集上最佳的单层决策树
    dataArr:特征标签
    classLabels:分类标签
    D:最初特征权重
    '''
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    #用于在特征的所有可能值上进行遍历
    numSteps=10.0
    #用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestStump={}
    bestClassEst=mat(zeros((m,1)))
    #无穷大
    minError=inf
    for i in range(n):
        #对每个特征
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            #每个步长
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0
                #整体结果的错误率
                weightedError=D.T*errArr
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=predictedVals.copy()
                    bestStump['dim']=i
                    #树的分界值
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClassEst
```

3. 基于单层决策树的AdaBoost训练过程

   AdaBoost算法：

   对每次迭代:

   &nbsp; &nbsp; &nbsp; &nbsp; 利用buildStump()找到最佳的单层决策树

   &nbsp; &nbsp; &nbsp; &nbsp; 将最佳单层决策树加入到单层决策树数组&nbsp; 

   &nbsp; &nbsp; &nbsp; &nbsp; 计算alpha

   &nbsp; &nbsp; &nbsp; &nbsp; 计算新的权重向量D

   &nbsp; &nbsp; &nbsp; &nbsp; 更新累计类别估计值

   &nbsp; &nbsp; &nbsp; &nbsp; 如果错误率等于0.0,退出循环

   ```python
   def adaBoostTrainDS(dataArr,classLabels,numIt=40):
       '''
       dataArr:特征标签
       classLabels:分类标签
       numIt:迭代次数
       '''
       #弱分类器集合
       weakClassArr=[]
       m=shape(dataArr)[0]
       D=mat(ones((m,1))/m)
       #预测分类结果值
       aggClassEst=mat(zeros((m,1)))
       for i in range(numIt):
           bestStrump,error,classEst=buildStump(dataArr,classLabels,D)
           print("D: ",D.T)
           #计算每个分类器的alpha权重值
           alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
           bestStrump['alpha']=alpha
           weakClassArr.append(bestStrump)
           print("classEst: ",classEst.T)
           #更新权重D
           expon=multiply(-1*alpha*mat(classLabels).T,classEst) 
           D=multiply(D,exp(expon))
           D=D/D.sum()
           #错误率累加
           aggClassEst+=alpha*classEst
           print("aggClassEst: ",aggClassEst.T)
           aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
           errorRate=aggErrors.sum()/m
           print("total error: ",errorRate,"\n")
           if errorRate==0.0:
               break
       return weakClassArr,aggClassEst
   ```

4. 分类函数

   ```python
   def adaClassify(datToClass,classifierArr):
       '''
       datToClass:多个待分类的样例
       classifierArr:多个弱分类器组成的数组
       '''
       dataMatrix=mat(datToClass)
       m=shape(dataMatrix)[0]
       aggClassEst=mat(zeros((m,1)))
       for i in range(len(classifierArr[0])):
           classEst=stumpClassify(dataMatrix,classifierArr[0][i]['dim'],classifierArr[0][i]['thresh'],classifierArr[0][i]['ineq'])
           aggClassEst+=classifierArr[0][i]['alpha']*classEst
           print(aggClassEst)
       return sign(aggClassEst)
   ```

   

[AdaBoost完整代码](https://github.com/TonyJent/myMachineLearning/blob/master/07_Ensemble%20Learning/adaboost.py)

[随机森林完整代码](https://github.com/TonyJent/myMachineLearning/blob/master/07_Ensemble%20Learning/RandomForest.py)

