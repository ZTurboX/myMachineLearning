from numpy import *
from bs4 import BeautifulSoup
import LinearRegression as LR

def scrapePage(retX,retY,inFile,yr,numPce,origPrc):
    '''
    从页面读取数据，生成retX,retY列表
    '''
    fr=open(inFile)
    soup=BeautifulSoup(fr.read())
    i=1
    #根据html页面结构进行解析
    currentRow=soup.findAll('table',r="%d" % i)
    while len(currentRow)!=0:
        currentRow=soup.findAll('table',r="%d" % i)
        title=currentRow[0].findAll('a')[1].text
        lwrTitle=title.lower()
        #查找是否有全新标签
        if(lwrTitle.find('new')>-1) or (lwrTitle.find('nisb')>-1):
            newFlag=1.0
        else:
            newFlag=0.0
        #查找是否已经标志出售
        soldUnicde=currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print("item #%d did not sell" % i)
        else:
            #解析页面获取当前价格
            soldPrice=currentRow[0].findAll('td')[4]
            priceStr=soldPrice.text
            priceStr=priceStr.replace('$','')
            priceStr=priceStr.replace(',','')
            if len(soldPrice)>1:
                priceStr=priceStr.replace('Free shipping','')
            sellingPrice=float(priceStr)

            #去掉不完整的套装价格
            if sellingPrice>origPrc*0.5:
                print("%d\t%d\t%d\t%f" % (yr,numPce,newFlag,origPrc,sellingPrice))
                retX.append([yr,numPce,newFlag,origPrc])
                retY.append(sellingPrice)
        i+=1
        currentRow=soup.findAll('table',r="%d" % i)
def setDataCollect(retX, retY):
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'input/8.Regression/setHtml/lego10196.html', 2009, 3263, 249.99)

def crossValidation(xArr,yArr,numVal=10):
    '''
    交叉验证测试岭回归
    '''
    m=len(yArr)
    indexList=range(m)
    errorMat=zeros((numVal,30))
    #交叉验证循环
    for i in range(numVal):
        #随机拆分数据
        trainX=[];trainY=[]
        testX=[];testY=[]
        #对数据混洗操作
        random.shuffle(indexList)
        for j in range(m):
            if j<m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        #回归系数矩阵
        wMat=LR.ridgeTest(trainX,trainY)
        #循环遍历矩阵中的30组回归系数
        for k in range(30):
            matTestX=mat(testX)
            matTrainX=mat(trainX)
            #数据标准化
            meanTrain=mean(matTrainX,0)
            varTrain=var(matTrainX,0)
            matTestX=(matTestX-meanTrain)/varTrain
            #测试回归效果
            yEst=matTestX*mat(wMat[k,:]).T+mean(trainY)
            #计算误差
            errorMat[i,k]=((yEst.T.A-array(testY))**2).sum()
    #计算误差估计值的均值
    meanErrors=mean(errorMat,0)
    minMean=float(min(meanErrors))
    bestWeights=wMat[nonzero(meanErrors==minMean)]
    #数据还原
    xMat=mat(xArr)
    yMat=mat(yArr).T
    meanX=mean(xMat,0)
    varX=var(xMat,0)
    unReg=bestWeights/varX

    print("teh best model from ridge regression is:\n",unReg)
    print("with constant term: ",-1*sum(multiply(meanX,unReg))+mean(yMat))

def regression():
    lgX=[]
    lgY=[]
    setDataCollect(lgX,lgY)
    crossValidation(lgX,lgY,10)