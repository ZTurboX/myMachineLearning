import random
from numpy import *
import svmMLiA2

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

def loadImages(dirName):
    from os import listdir
    hwLabels=[]
    trainingFileList=listdir(dirName)
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        if classNumStr==9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:]=img2vector('%s/%s' % (dirName,fileNameStr))
    return trainingMat,hwLabels


def testDigits(kTup=('rbf',10)):
    #导入训练数据
    dataArr,labelArr=loadImages('trainingDigits')
    b,alphas=svmMLiA2.smoP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat,labelMat=mat(dataArr),mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print("there are %d support vectors" % shape(sVs)[0])
    m,n=shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=svmMLiA2.kernelTrans(sVs,dataMat[i,:],kTup)
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr=loadImages('testDigits')
    errorCount=0
    dataMat,labelMat=mat(dataArr),mat(labelArr).transpose()
    m,n=shape(dataMat)
    for i in range(m):
        kernelEval=svmMLiA2.kernelTrans(sVs,dataMat[i,:],kTup)
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelArr[i]):
            errorCount+=1
    print("the test error rate is: %f" % (float(errorCount)/m))


if __name__=='__main__':
    testDigits(('rbf',20))