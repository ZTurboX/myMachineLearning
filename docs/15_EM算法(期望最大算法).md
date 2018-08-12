# EM算法(期望最大算法)

## 概述

EM算法是常用的估计参数隐变量的利器，它是一种迭代式的方法，其基本想法是：若参数已知，则可根据训练数据推断出最优隐变量的值；若最优隐变量的值已知，则可对参数做最大似然估计

## Jensen不等式

若f是定义域为实数的函数，如果对于所有实数x，f''(x)>=0，那么f是凸函数。当x是向量时，如果其hession矩阵是H是半正定的(H>=0)，那么f是凸函数。如果f''(x)>0或者H>0，那么f是严格的凸函数。

Jensen不等式:

如果f是凸函数，X是随机变量，那么E[f(X)]>=f(EX)

如果f是严格凸函数，那么E[f(X)]=f(EX)当且仅当p(x=E[X])=1

如图：

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/Jensen%E4%B8%8D%E7%AD%89%E5%BC%8F.PNG)

## EM算法

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/EM%E7%AE%97%E6%B3%95E%E6%AD%A5.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/EM%E7%AE%97%E6%B3%95M%E6%AD%A51.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/EM%E7%AE%97%E6%B3%95M%E6%AD%A52.PNG)

## EM算法在高斯混合模型中的应用

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/%E9%AB%98%E6%96%AF%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B1.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/%E9%AB%98%E6%96%AF%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B2.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/%E9%AB%98%E6%96%AF%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B3.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/%E9%AB%98%E6%96%AF%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B4.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/%E9%AB%98%E6%96%AF%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B5.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/%E9%AB%98%E6%96%AF%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B6.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/15_EM/%E9%AB%98%E6%96%AF%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B7.PNG)

## 高斯混合模型算法实现

```python
from numpy import *

def initData(k,u,sigma,dataNum):
    '''
    初始化高斯混合模型的数据
    k:比例系数
    u:均值
    sigma:标准差
    dataNum:数据个数
    '''
    dataSet=zeros(dataNum,dtype=float)
    #高斯分布个数
    n=len(k)
    for i in range(dataNum):
        #产生0-1的随机数
        rand=random.random()
        sK=0
        index=0
        while index<n:
            sK+=k[index]
            if rand<sK:
                dataSet[i]=random.normal(u[index],sigma[index])
                break
            else:
                index+=1
    return dataSet


def normalFun(x,u,sigma):
    '''
    计算均值为u,标准差为sigma的正太分布函数的密度函数值
    '''
    return (1.0/sqrt(2*pi)*sigma)*(exp(-(x-u)**2/(2*sigma**2)))

def em(dataSet,k,u,sigma,step=10):
    '''
    高斯混合模型
    '''
    n=len(k)
    dataNum=len(dataArr)
    gamaArr=zeros((n,dataNum))
    for s in range(step):
        #E步，确定Q函数
        for i in range(n):
            for j in range(dataNum):
                wSum=sum([k[t]*normalFun(dataSet[j],u[t],sigma[t]) for t in range(n)])
                gamaArr[i][j]=k[i]*normalFun(dataSet[j],u[i],sigma[i])/float(wSum)
        
        #M步
        #更新u
        for i in range(n):
            u[i]=sum(gamaArr[i]*dataSet)/sum(gamaArr[i])
        #更新sigma
        for i in range(n):
            sigma[i]=sqrt(sum(gamaArr[i]*(dataSet-u[i])**2)/sum(gamaArr[i]))
        #更新k
        for i in range(n):
            k[i]=sum(gamaArr[i])/dataNum
    
    return [k,u,sigma]
```

