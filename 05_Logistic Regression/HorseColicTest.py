from numpy import *
import matplotlib.pyplot as plt
import LogisticRegression as LR 

def classifyVector(inX,weights):
    '''
    LogisticRegression分类
    inX:特征向量
    weights:回归系数
    '''
    prob=LR.sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    '''
    对数据格式化处理
    '''
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=LR.stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0
    numTestVec=0.0
    #读取测试集数据，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the error rate is: %f" % errorRate)
    return errorRate

def multiTest():
    '''
    调用colicTest()10次求平均值
    '''
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests)))


if __name__=='__main__':
   multiTest()
