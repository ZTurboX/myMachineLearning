from numpy import *
import operator
from os import listdir


def classify0(inX,dataSet,labels,k):
    '''
    intX:用于分类的输入向量
    dataSet:输入的训练样本集
    labels:标签向量
    k:用于选择最近邻居的数目
    '''
    #1.距离计算
    dataSetSize=dataSet.shape[0]
    #生成和训练样本对应的矩阵，并与训练样本求差
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2#取平方
    #将矩阵每一行相加
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5#开方
    #距离排序,取其索引
    sortedDistIndicies=distances.argsort()

    #2.选择距离最小的k个点
    classCount={}
    for i in range(k):
        #找到该样本的类型
        voteIlabel=labels[sortedDistIndicies[i]]
        #在字典中将该类型加一
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #3.排序并返回出现最多的类型
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
    '''
    将图像转化为测试向量
    '''
    #创建1*1024的矩阵
    returnVect=zeros((1,1024))
    fr=open(filename)
    #循环读文件的前32行
    for i in range(32):
        lineStr=fr.readline()
        #将每行前32个字符存储在数组中
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest():
    '''
    使用k-近邻算法识别手写数字
    '''
    #类别向量
    hwLabels=[]
    #获取目录的文件名，将目录中的文件内容存储在列表中
    trainingFileList=listdir('trainingDigits')
    #获取目录中有多少文件
    m=len(trainingFileList)
    #创建m*1024的训练矩阵
    trainingMat=zeros((m,1024))
    #从文件名解析分类数字
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNameStr=int(fileNameStr.split('_')[0])
        hwLabels.append(classNameStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr)
    
    testFileList=listdir('testDigits')
    errorCount=0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d, the real answer is: %d"%(classifierResult,classNumStr))
        if classifierResult!=classNumStr:
            errorCount+=1
    print("\nthe total number of errors is: %d"%errorCount)
    print("\nthe total error rate is: %f"%(errorCount/float(mTest)))        



