
from numpy import *
import operator

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


def file2matrix(filename):
    '''
    1.准备数据，从文本文件解析数据
    返回训练样本矩阵和类标签向量
    '''
    fr=open(filename)
    #获取文件中的数据的行数
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    #生成一个numberOfLines*3的矩阵，全置0
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        #截取所有回车字符
        line=line.strip()
        #使用\t将上一步得到的整行数据分割成元素列表
        listFromLine=line.split('\t')
        #每列的属性数据
        returnMat[index,:]=listFromLine[0:3]
        #每列的类别数据
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector


def autoNorm(dataSet):
    '''
    2.归一化数据
    newValue=(oldValue-min)/(max-min)
    '''
    #计算每种属性的最大值，最小值
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    #极差
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    #生成与最小值之差组成的矩阵
    normDataSet=dataSet-tile(minVals,(m,1))
    #将最小值之差除以范围组成的矩阵
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    '''
    kNN分类器针对约会网站的测试代码
    '''
    hoRatio=0.10#测试集范围
    #从文件加载数据
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    #归一化数据
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    #设置测试样本数量
    numTestVexs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVexs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVexs:m,:],datingLabels[numTestVexs:m],3)
        print("the classifier came back with: %d, the real answer is: %d"%(classifierResult,datingLabels[i]))
        if classifierResult!=datingLabels[i]:
            errorCount+=1
    print("the total error rate is: %f" % (errorCount/float(numTestVexs)))
