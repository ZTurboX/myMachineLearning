from numpy import *
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[list(map(float,line)) for line in stringArr]
    return mat(datArr)

'''
PCA算法：
    1. 去除平均值
    2. 计算协方差矩阵
    3. 计算协方差矩阵的特征值和特征向量
    4. 将特征值从大到小排序
    5. 保留最上面的N个特征向量
    6. 将数据转换到上述N个特征向量构建的新空间中
'''

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

if __name__=='__main__':
    dataMat=loadDataSet('testSet.txt')
    lowDMat,reconMat=pca(dataMat,2)
    # print(dataMat)
    # print("-------------------------------------------")
    # print(lowDMat)
    # print("-------------------------------------------")
    # print(reconMat)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
    plt.show()

