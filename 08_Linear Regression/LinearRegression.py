from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    #特征值
    dataMat=[]
    #标签值
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        #删除一行中以tab分割的数据前后的空白符号
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    '''
    标准线性回归
    '''
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    #linalg.det()用来求矩阵行列式，如果行列式为0，矩阵不可逆
    if linalg.det(xTx)==0.0:
        print("this matrix is singular,cannot do inverse")
        return 
    #最小二乘法,求回归系数
    ws=xTx.I*(xMat.T*yMat)
    return ws


def lwlr(testPoint,xArr,yArr,k=1.0):
    '''
    局部加权线性回归，在待测点附近的每个点赋予一定的权重
    回归系数=(X^T*W*X)*I*X^T*W*y
    权重=exp(|x(i)-x|/((-2)*k^2))
    '''
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    #创建一个对角线元素为1，其余元素为0的二维数组
    weights=mat(eye((m)))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print("this matrix is singular,cannot do inverse")
        return 
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws
    
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

def regression2():
    '''
    局部加权回归
    '''
    xArr,yArr=loadDataSet('data.txt')
    yHat=lwlrTest(xArr,xArr,yArr,0.003)
    xMat=mat(xArr)
    #从小到达排序，提取index
    srtInd=xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].tolist(),mat(yArr).T.tolist(),s=2,c='red')
    plt.show()
    

def ridgeRegres(xMat,yMat,lam=0.2):
    '''
    岭回归:用于特征数比样本点多的数据集
    '''
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print("this matrix is singular,cannot do inverse")
        return 
    ws=denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    
    #计算y的均值
    yMean=mean(yMat,0)
    #标准化y
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    #x的方差
    xVar=var(xMat,0)
    #x归一化
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def regularize(xMat):
    inMat=xMat.copy()
    inMeans=mean(inMat,0)
    inVar=var(inMat,0)
    inMat=(inMat-inMeans)/inVar
    return inMat


def stageWise(xArr,yArr,eps=0.01,numIt=100):
    '''
    向前逐步线性回归：

    数据标准化，使其分布满足0均值和单位方差
    在每轮迭代中：
        设置当前最小误差lowestError为正无穷
        对每个特征：
            增大或缩小：
                改变一个系数得到一个新的w
                计算新w下的误差
                如果误差error小于当前最小误差lowestError，设置Wbest为当前的W
            将W设置为新的Wbest
    '''
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    xMat=regularize(xMat)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    wsMax=ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError=inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat


def regression4():
    '''
    向前逐步回归
    '''
    xArr,yArr=loadDataSet('abalone.txt')
    print(stageWise(xArr,yArr,0.01,200))
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xMat=regularize(xMat)
    yM=mean(yMat,0)
    yMat=yMat-yM
    weights=standRegres(xMat,yMat.T)
    print(weights.T)

def regression3():
    '''
    岭回归
    '''
    abX,abY=loadDataSet('abalone.txt')
    ridgeWeights=ridgeTest(abX,abY)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


def regression1():
    '''
    标准线性回归
    '''
    xArr,yArr=loadDataSet('data.txt')
    ws=standRegres(xArr,yArr)
    xMat=mat(xArr)
    yMat=mat(yArr)
    yHat=xMat*ws
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].tolist(),yMat.T[:,0].tolist())
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

regression4()



