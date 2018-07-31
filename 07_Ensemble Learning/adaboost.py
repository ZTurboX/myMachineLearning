from numpy import *

def loadSimpData():
    datMat=matrix([[1. ,2.1],
                   [2. ,1.1],
                   [1.3,1. ],
                   [1. ,1. ],
                   [2. ,1. ]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

'''
单层决策树算法：
将最小错误率minError设为正无穷大
对数据集中的每个特征：
    对每个步长：
        对每个不等号:
            建立一棵单层决策树并利用加权数据集对它测试
            如果错误率低于minError,则将当前单层决策树设为最佳单层决策树
返回最佳单层决策树
'''

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


'''
AdaBoost算法：
对每次迭代:
    利用buildStump()找到最佳的单层决策树
    将最佳单层决策树加入到单层决策树数组
    计算alpha
    计算新的权重向量D
    更新累计类别估计值
    如果错误率等于0.0,退出循环
'''

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


if __name__=='__main__':
    datMat,classLabels=loadSimpData()
    classifierArray=adaBoostTrainDS(datMat,classLabels,30)
    print(classifierArray)
    print(adaClassify([0,0],classifierArray))
    print(classifierArray[0])
