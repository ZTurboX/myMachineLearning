'''
kNN原理:
1.假设有一个带有标签的样本数据集，其中包含每条数据与所属分类的对应的关系
2.输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较
    a.计算新数据与样本数据集中每条数据的距离
    b.对求得的所有距离进行排序
    c.取前k个样本数据对应的分类标签
3.求k个数据中出现次数最多的分类标签作为新数据的分类
'''

from numpy import *
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

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

if __name__=='__main__':
    group,labels=createDataSet()
    res=classify0([0,0],group,labels,3)
    print(res)



