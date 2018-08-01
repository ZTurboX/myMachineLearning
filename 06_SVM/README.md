# 支持向量机

## 概述

支持向量机是一种监督学习算法

## 基本理论

### 支持向量和划分超平面

将数据集分割开来的直线为划分超平面，划分超平面位于两类训练样本正中间，如图红色直线

支持向量是离划分超平面最近的那些点，如图红色圈

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F.PNG)

### 支持向量机基本型推导

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E9%97%B4%E9%9A%94.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%E5%9F%BA%E6%9C%AC%E5%9E%8B.PNG)

### 对偶问题

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/%E5%AF%B9%E5%81%B6%E9%97%AE%E9%A2%981.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/%E5%AF%B9%E5%81%B6%E9%97%AE%E9%A2%982.PNG)

### 序列最小优化算法（SMO）

#### 核心思想

创建一个alpha向量并将其初始化为0向量

当迭代次数小于最大迭代次数时：

&nbsp; &nbsp; &nbsp; &nbsp; 对数据集中的每个数据向量：

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 如果该数据向量可以被优化：

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 随机选择另外一个数据向量&nbsp; 

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 同时优化这两个向量

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 如果两个向量都不能被优化，退出内循环&nbsp; &nbsp; &nbsp; 

&nbsp; &nbsp; &nbsp; &nbsp;如果所有向量都没被优化，增加迭代数目，继续下一次循环

#### 公式推导（摘自《统计学习方法》）

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO1.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO2.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO3.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO4.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO5.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO6.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO7.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO8.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO9.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO10.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO11.PNG)

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/SMO12.PNG)

### 核函数

对于非线性，需要用到核函数，将数据转换为分类器。即可以将数据从某个特征空间到另一个特征空间的映射。

经过空间转换后，低维需要解决的非线性问题，就变为了高维需要解决的线性问题。核函数将所有的运算都可以写成内积的形式。

常用核函数：

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/06_SVM/%E6%A0%B8%E5%87%BD%E6%95%B0.PNG)

## 核心算法

### 简化版SMO算法

对文件解析

```python
import random
from numpy import *
def loadDataSet(fileName):
    '''
    对文件解析
    '''
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
```

```python
def selectJrand(i,m):
    '''
    i:第一个alpha下标
    m:所有alpha数目
    '''
    j=i
    #只要函数值不等于输入值i，函数就会随机选择
    while j==i:
        j=int(random.uniform(0,m))
    return j
```

调整alpha值

```python
def clipAlpha(aj,H,L):
    '''
    用于调整大于H或小于L的alpha值
    '''
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj
```

smo算法

```python
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    '''
    dataMatLn:特征集合
    classLabels:类别标签
    C:松弛常量，允许有些数据点可以处于分割面的错误一侧
                控制最大化间隔，保证大部分的函数间隔小于1.0这两个目标权重
    toler:容错率：某个体系中能减小一些因素或选择对某个系统产生不稳定的概率
    maxIter:退出最大的循环次数

    return:
        b:模型的常量值
        alphas:拉格朗日乘子
    '''
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)

    b=0
    alphas=mat(zeros((m,1)))
    #没有任何alpha改变的情况下遍历数据的次数
    iter=0
    while iter<maxIter:
        #记录alpha是否已经优化，每次循环设为0，然后再对整个集合顺序遍历
        alphaPairsChanged=0
        for i in range(m):
            #f=w^T*x[i]+b=Σa[n]*label[n]*x[n]*x[i]+b
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            #预测结果与真实结果的误差
            Ei=fXi-float(labelMat[i])
        
            '''
            检验样本(xi,yi)是否满足KKT条件：
            yi*f(i) >= 1 and alpha=0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha<C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the  boundary)

            发生错误的概率labelMat[i]*Ei如果超出了toler,才需要优化
            '''
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                #如果满足优化条件，随机选取非i的点，进行优化比较
                j=selectJrand(i,m)
                #预测j的结果
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()

                #L,H用于将alphas[j]调整到0-C之间。如果L==H就不做改变
                #labelMat[i]!=labelMat[j]表示异侧，就相减，否则是同侧，相加
                if labelMat[i]!=labelMat[j]:
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                #如果相同，就不优化
                if L==H:
                    print("L==H")
                    continue

                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print("eta>=0")
                    continue
                
                #计算新的alphas[j]值
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                #用L和H对其调整
                alphas[j]=clipAlpha(alphas[j],H,L)
                #检查alphas[j]是否是轻微的改变，如果是退出循环
                if abs(alphas[j]-alphaJold)<0.00001:
                    print("j not moving enough")
                    continue
                #对alphas[i]同样进行改变
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

                #计算阈值b
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T

                if 0<alphas[i] and C>alphas[i]:
                    b=b1
                elif 0<alphas[i] and C>alphas[j]:
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if alphaPairsChanged==0:
            iter+=1
        else:
            iter=0
        print("iteration number: %d" % iter)
    return b,alphas
```

[完整版SMO算法](https://github.com/TonyJent/myMachineLearning/blob/master/06_SVM/svmMLiA2.py)

## 项目实战

[手写识别](https://github.com/TonyJent/myMachineLearning/blob/master/06_SVM/handWritingClassify2.py)

