# Logistic回归

## 概述

Logistic回归虽然叫做回归，但还属于分类，主要思想：根据现有数据对分类边界线建立回归公式，以此进行分类。

## 基本原理

### Sigmoid函数

我们想要的函数是能接受所有的输入然后预测出类别，sigmoid函数就有这样的性质

sigmoid函数：

![sigmoid函数](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/sigmoid%E5%87%BD%E6%95%B0%E5%85%AC%E5%BC%8F.PNG)

sigmoid函数图像：

![sigmoid函数图像](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/sigmoid%E5%87%BD%E6%95%B0%E5%9B%BE.jpg)

根据图像我们可以看到：当x=0时，sigmoid函数值为0.5；随着x的增大，函数值逼近于1，随着x的减小，函数值逼近于0。

为了实现Logistic回归分类器，我们可以在每个特征上都乘以一个回归系数，然后把所有的结果值相加，将这个总和代入sigmoid函数，进而得到一个范围在0~1之间的数值。如果数据大于0.5被分入1类，小于0.5分为0类。

记sigmoid函数输入为z,则

![输入z](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/%E8%BE%93%E5%85%A5z.PNG)

其中x为输入数据，w为最佳拟合系数

## 梯度上升法

主要思想：找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻。函数f(x,y)的梯度：

![f(x,y)的梯度](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/f(x%2Cy)%E6%A2%AF%E5%BA%A6.PNG)

梯度意味着要沿x![x方向移动](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/%E6%B2%BFx%E6%96%B9%E5%90%91%E7%A7%BB%E5%8A%A8.PNG) ,沿y![y方向移动](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/%E6%B2%BFy%E6%96%B9%E5%90%91%E7%A7%BB%E5%8A%A8.PNG)

梯度算法总是指向函数值增长最快的方向，移动方向的量值称为步长，记为α ，梯度上升算法迭代公式:

![梯度上升算法迭代公式](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/%E6%A2%AF%E5%BA%A6%E4%B8%8A%E5%8D%87%E6%B3%95%E8%BF%AD%E4%BB%A3%E5%85%AC%E5%BC%8F.PNG)

## 核心算法

### 梯度上升优化算法

1. 载入数据集

   ```python
   from numpy import *
   import matplotlib.pyplot as plt
   
   def loadDataSet():
       dataMat=[]
       labelMat=[]
       fr=open('testSet.txt')
       for line in fr.readlines():
           lineArr=line.strip().split()
           dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
           labelMat.append(int(lineArr[2]))
       return dataMat,labelMat
   ```

2. 计算sigmoid函数值

   ```python
   def sigmoid(inX):
       return 1.0/(1+exp(-inX))
   ```

3. 梯度上升算法

   梯度上升思想：

   ​	每个回归系数初始化为1

   ​	重复R次：

   ​		计算整个数据集的梯度

   ​		使用alpha*gradient更新回归系数的向量

   ​	返回回归系数

   ```python
   def gradAscent(dataMatIn,classLabels):
       '''
       Logistic回归梯度上升优化算法
       dataMatIn:2维numpy矩阵，每列代表每个不同的特征，每行代表每个训练样本
       classLabels:类别标签
       '''
       #转换为numpy矩阵
       dataMatrix=mat(dataMatIn)
       #转化为numpy矩阵，再将行向量转换为列向量
       labelMat=mat(classLabels).transpose()
       #m:数据量，n特征数
       m,n=shape(dataMatIn)
       #alpha为目标移动的步长
       alpha=0.001
       #迭代次数
       maxCyles=500
       #生成一个长度和特征数相同的矩阵
       #weigths为回归系数
       weights=ones((n,1))
       for k in range(maxCyles):
           #预测值
           h=sigmoid(dataMatrix*weights)
           #真实类别与预测类别的差值
           error=(labelMat-h)
           weights=weights+alpha*dataMatrix.transpose()*error
       return array(weights)
   ```

   

代码weights=weights+alpha*dataMatrix.transpose()*error的推导：

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/%E6%8E%A8%E5%AF%BC%E5%85%AC%E5%BC%8F.PNG)

4. 画出决策边界

   ```python
   def plotBestFit(weights):
       '''
       画出决策边界
       weights:回归系数
       '''
       dataMat,labelMat=loadDataSet()
       dataArr=array(dataMat)
       n=shape(dataArr)[0]
       xcord1=[]; ycord1=[]
       xcord2=[]; ycord2=[]
       for i in range(n):
           if int(labelMat[i])==1:
               xcord1.append(dataArr[i,1])
               ycord1.append(dataArr[i,2])
           else:
               xcord2.append(dataArr[i,1])
               ycord2.append(dataArr[i,2])
       fig=plt.figure()
       ax=fig.add_subplot(111)
       ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
       ax.scatter(xcord2,ycord2,s=30,c='green')
       x=arange(-3.0,3.0,0.1)
       '''
       z=w0*x0+w1*x1+w2*x2
       x0=1,x1=x,x2=y,令z=0:
       w0+w1*x+w2*y=0-->y=(-w0-w1*x)/w2
       '''
       y=(-weights[0]-weights[1]*x)/weights[2]
       ax.plot(x,y)
       plt.xlabel('X')
       plt.ylabel('Y')
       plt.show()
   ```

   

梯度上升优化算法拟合直线

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/%E6%A2%AF%E5%BA%A6%E4%B8%8A%E5%8D%87%E4%BC%98%E5%8C%96%E6%9C%80%E4%BD%B3%E6%8B%9F%E5%90%88%E7%9B%B4%E7%BA%BF.PNG)

### 梯度上升算法改进

如果数据集变得很大，梯度上升优化算法需要大量计算，我们可以使用随机梯度上升算法

随机梯度上升：是一个在线学习算法，一次处理所有数据称为“批处理”

​    思想：

​        所有回归系数初始化为1

​        对数据集中每个样本：

​            计算该样本的梯度

​            使用alpha*gradient更新回归系数值

```python
def stocGradAscent0(dataMatrix,classLabels):
    '''
    随机梯度上升：是一个在线学习算法，一次处理所有数据称为“批处理”
    算法：
        所有回归系数初始化为1
        对数据集中每个样本：
            计算该样本的梯度
            使用alpha*gradient更新回归系数值
    '''
    m,n=shape(dataMatrix)
    weights=ones(n)
    alpha=0.01
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=(classLabels[i]-h)
        weights=weights+alpha*error*dataMatrix[i]
    return weights

```

 拟合直线

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8A%E5%8D%87%E6%8B%9F%E5%90%88%E7%9B%B4%E7%BA%BF.PNG)

随机梯度上升算法在每次迭代时会引发系数的剧烈改变，我们期望算法能避免来回波动，从而收敛到某个值，我们来改进随机梯度上升算法

```python
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    '''
    随机梯度上升算法改进
    '''
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            #alpha在每次迭代调整，i和j不断增大，alpha不断减少，但不会减小到0
            alpha=4/(1.0+j+i)+0.0001
            #随机选取样本更新回归系数
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
```

拟合直线

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/05_LogisticRegression/%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8A%E5%8D%87%E6%94%B9%E8%BF%9B%E6%8B%9F%E5%90%88%E7%9B%B4%E7%BA%BF.PNG)

## 项目实战

[ 从疝气病症预测病马的死亡率](https://github.com/TonyJent/myMachineLearning/blob/master/05_LogisticRegression/HorseColicTest.py)



