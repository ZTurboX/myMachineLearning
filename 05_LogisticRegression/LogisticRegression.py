'''
梯度上升法：
每个回归系数初始化为1
重复R次：
    计算整个数据集的梯度
    使用alpha*gradient更新回归系数的向量
返回回归系数
'''
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

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

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



if __name__=='__main__':
    dataArr,labelMat=loadDataSet()
    #weights=gradAscent(dataArr,labelMat)
    #weights=stocGradAscent0(array(dataArr),labelMat)
    weights=stocGradAscent1(array(dataArr),labelMat)
    plotBestFit(weights)

