'''
朴素贝叶斯工作原理:

提取所有文档中的词条并去重
获取文档的所有类别
计算每个类别中的文档数目
对每篇文档：
    对每个类别：
        如果词条出现在文档中-->增加该词条的计数值
        增加所有词条的计数值（此类别下词条总数）
对每个类别：
    对每个词条：
        将该词条的数目除以总词条数目得到的条件概率(P(词条|类别))
返回该文档属于每个类别的条件概率(P(类别|文档的所有词条))
'''
from numpy import *

def loadDataSet():
    '''
    创建数据集
    '''
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #1代表侮辱性文字，0代表正常言论
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    '''
    获取所有单词的集合
    '''
    vocabSet=set([])
    for document in dataSet:
        #创建两个集合的并集
        vocabSet=vocabSet | set(document)
    return list(vocabSet)

def setWords2Vec(vocabList,inputSet):
    '''
    遍历查看该单词是否出现，出现该单词则将该单词置1
    vocabList:所有单词集合列表
    inputSet:输入数据集
    '''
    #创建一个和词汇表等长的向量，元素置为0
    returnVec=[0]*len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word:%s is not in my vocabulary"%word)
    return returnVec

def bagOfWords2VecMN(vocabList,inputSet):
    '''
    朴素贝叶斯词袋模型
    '''
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
    return returnVec
    

def trainNB0(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类训练函数：
    trainMatrix:文件单词矩阵
    trainCategory:文件对应的类别
    '''
    #文件数
    numTrainDocs=len(trainMatrix)
    #单词数
    numWords=len(trainMatrix[0])
    #计算侮辱性文件的出现的概率，trainCategory所有1的求和/文件总数
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #构造单词出现次数列表
    #p0Num=zeros(numWords)
    #p1Num=zeros(numWords)
    #整个数据集单词出现总数
    #p0Denom=0.0
    #p1Denom=0.0
    '''
    利用朴素贝叶斯对文档分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，
    即计算p(w0|1)p(w1|1)p(w2|1)
    如果其中一个概率为0，那么最后乘积也为0.所以将词出现数初始化为1，分母初始化为2
    '''
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
   
    for i in range(numTrainDocs):
        #是否是侮辱性文件
        if trainCategory[i]==1:
            #如果是侮辱性文件，对侮辱性文件的向量相加
            p1Num+=trainMatrix[i] 
            #对向量中的所有元素求和，计算所有侮辱性文件中出现的单词总数
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    
    #在类别1下每个单词出现的概率
    #p1Vect=p1Num/p1Denom
    #在类别0下每个单词出现的概率
    #p0Vect=p0Num/p0Denom

    '''
    由于太多很小的数相乘，会造成下溢，则将p1Vect,p0Vect分别取对数
    '''
    #在类别1下每个单词出现的概率
    p1Vect=log(p1Num/p1Denom)
    #在类别0下每个单词出现的概率
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive



def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''
    朴素贝叶斯分类函数:将乘法转换为加法
    乘法:P(C|F1F2....Fn)=P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
    加法:P(F1|C)*P(F2|C)...P(Fn|C)P(C)-->log(P(F1|C))+log(P(F2|c))....+log(P(Fn|C))+log(P(C))
    
    vec2Classify:待测数据，要分类的向量
    p0Vec:类别0，[log(P(F1|C0)),log(P(F2|C0))....log(P(Fn|C0))]列表
    p1Vec:类别1，[log(P(F1|C1)),log(P(F1|C1))....log(P(F1|C1))]列表
    pClass1:类别1，侮辱性文件出现的概率
    '''
    #计算log(P(F1|C))+log(P(F2|c))....+log(P(Fn|C))+log(P(C))
    p1=sum(vec2Classify*p1Vec)+log(pClass1)#P(w|c1)*P(c1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)#P(w|c0)*P(c0)
    if p1>p0:
        return 1
    else:
        return 0


def testingNB():
    '''
    测试算法
    '''
    #加载数据
    listOPosts,listClasses=loadDataSet()
    #创建单词集合
    myVocabList=createVocabList(listOPosts)
    #计算单词是否出现并创建数据矩阵
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setWords2Vec(myVocabList,postinDoc))
    #训练数据
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    #测试数据
    testEntry=['love','my','dalmation']
    thisDoc=array(setWords2Vec(myVocabList,testEntry))
    print(myVocabList)
    print(thisDoc)
    print(p0V)
    print(p1V)
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=array(setWords2Vec(myVocabList,testEntry))
    #print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))



if __name__=='__main__':
    testingNB()




# if __name__=='__main__':
#     listOPosts,listClasses=loadDataSet()
#     myVocabList=createVocabList(listOPosts)
#     print(myVocabList)
#     trainMat=[]
#     for postinDoc in listOPosts:
#         trainMat.append(setWords2Vec(myVocabList,postinDoc))
#     print(trainMat)
#     p0V,p1V,pAb=trainNB0(trainMat,listClasses)
#     print(pAb)
#     print(p1V)
#     print(p0V)