import bayes
import random
from numpy import *

def textParse(bigStirng):
    '''
    切分文本：去掉少于2个字符的字符串，并将所有字符串转换成小写，返回字符串列表
    '''
    import re
    #分割除单词，数字外的任意字符串
    listOfTokens=re.split(r'\W*',bigStirng)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    '''
    对贝叶斯垃圾邮件分类器进行自动化处理
    对测试集中的每封邮件进行分类，若邮件分类错误，测错误数加1，最后返回总的错误百分比
    '''
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        #切分，解析数据，并归类为1的类别
        wordList=textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
    #创建词汇表
    vocabList=bayes.createVocabList(docList)
    trainingSet=list(range(50))
    testSet=[]
    #随机取10个邮件测试
    for i in range(10):
        #random.uniform(x,y)随机产生一个范围为x-y的实数
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bayes.setWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=bayes.trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    errorDoc=[]
    for docIndex in testSet:
        wordVector=bayes.setWords2Vec(vocabList,docList[docIndex])
        if bayes.classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
            errorDoc.append(docList[docIndex])
    print(vocabList)
    print(trainMat)
    print('the errorCount is: ',errorCount)
    print('the testSet length is: ',len(testSet))
    print('the error rate is； ',float(errorCount)/len(testSet))
    print(errorDoc)

if __name__=='__main__':
    spamTest()
