import bayes
import random
from numpy import *
import filterSpam
import feedparser

'''
RSS源分类器及高频词去除函数
'''
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict={}
    #遍历词汇表中的每个词
    for token in vocabList:
        #统计每个词在文本中出现的次数
        freqDict[token]=fullText.count(token)
    #根据每个词出现的次数从高到低排序
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    #返回出现次数最高的30个单词
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList=[]
    classList=[]
    fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        #每次访问一条RSS源
        wordList=filterSpam.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=filterSpam.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList) 
        classList.append(0)
    vocabList=bayes.createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        #去除出现次数最高的那些词
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet=list(range(2*minLen))
    testSet=[]
    for i in range(5):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=bayes.trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=bayes.bagOfWords2VecMN(vocabList,docList[docIndex])
        if bayes.classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]
    topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-6.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("Sf***********************************************SF")
    for item in sortedSF:
        print(item[0])
    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print("NY**********************************************NY")
    for item in sortedNY:
        print(item[0])




if __name__=='__main__':
    ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    sy=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
    getTopWords(ny,sy)