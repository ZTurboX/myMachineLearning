# 利用PCA来简化数据

## 降维技术

### 主成分分析(PCA)

在PCA中，数据 从原来的坐标系转换到了新的坐标系，新坐标系的选择是由数据本身决定的。第一个新坐标轴选择的是原来数据中方差最大的方向，第二个新坐标轴选择和第一个坐标轴正交且具有最大方差的方向。该过程一直重复，重复次数为原始数据中特征的数目。

### 因子分析

假设观察数据的成分中有一些观察不到的隐变量 ，假设观察数据是这些隐变量和某些噪音的线性组合，那么隐变量的数据可能比观察数据的数目少，也就说通过找到隐变量就可以实现数据的降维。

### 独立成分分析(ICA)

ICA 是假设数据是从 N 个数据源混合组成的，这一点和因子分析有些类似，这些数据源之间在统计上是相互独立的，而在 PCA 中只假设数据是不 相关（线性关系）的。同因子分析一样，如果数据源的数目少于观察数据的数目，则可以实现降维过程。

## PCA

### 原理

1. 去除平均值

​    2. 计算协方差矩阵

​    3. 计算协方差矩阵的特征值和特征向量

​    4. 将特征值从大到小排序

​    5. 保留最上面的N个特征向量

​    6. 将数据转换到上述N个特征向量构建的新空间中

### 公式

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/13_PCA/%E5%85%AC%E5%BC%8F.PNG)

### 算法

```python
def pca(dataMat,topNfeat=9999999):
    '''
    topNfeat:应用的N个特征
    '''
    #计算每一列的均值
    meanVals=mean(dataMat,axis=0)
    #去均值
    meanRemoved=dataMat-meanVals
    #计算协方差矩阵
    '''
    协方差cov(X,Y)=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+....+(xn-x均值)*(yn-y均值)]/(n-1)
    方差:(一维)度量两个随机变量关系的统计量
    协方差:(二维)度量各个维度偏离其均值的程度
    协方差矩阵:(n维)度量各个维度偏离其均值的程度
    '''
    covMat=cov(meanRemoved,rowvar=0)
    #计算特征值和特征向量
    eigVals,eigVects=linalg.eig(mat(covMat))
    #特征值从小到大排序，返回index序号
    eigValInd=argsort(eigVals)
    #返回topN的特征值，倒序
    eigValInd=eigValInd[:-(topNfeat+1):-1]
    #保存特征向量
    redEigVects=eigVects[:,eigValInd]
    #将数据转换到新空间
    lowDDataMat=meanRemoved*redEigVects#降维后的数据集
    reconMat=(lowDDataMat*redEigVects.T)+meanVals#新的数据集空间
    return lowDDataMat,reconMat
```

