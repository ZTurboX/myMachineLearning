from numpy import *


def loadDataSet(fileName):
    '''
    从文本文件构建矩阵
    '''
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    '''
    计算两个向量的欧式距离
    '''
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    '''
    构建一个包含k个随机质心的集合，质心要包含在数据集的边界内
    '''
    n=shape(dataSet)[1]
    #创建k个质心矩阵
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=mat(minJ+rangeJ*random.rand(k,1))
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    '''
    k-means算法：
    创建k个点作为起始质心
    当任意一个点的族分配结果发生改变时：
        对数据集中的每个数据点：
            对每个质心：
                计算质心与数据点间的距离
            将数据点分配到距其最近的族
        对每个族，计算族中所有点的均值并将均值作为质心
    '''
    m=shape(dataSet)[0]
    #创建族分配矩阵
    clusterAssment=mat(zeros((m,2)))
    #创建质心
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf
            minIndex=-1
            #寻找最近的质心
            for j in range(k):
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            #族分配结果改变
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
                clusterAssment[i,:]=minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment

def biKmeans(dataSet,k,distMeas=distEclud):
    '''
    二分k-means:
    将所有点看成一个族
    当族数目小于k时：
        对于每个族：
            计算总误差
            在给定的族上面进行kMeans聚类
            计算将该族一分为二后的总误差
        选择使得误差最小的那个族进行划分操作
    '''
    m=shape(dataSet)[0]
    #保存每个数据点的族分配结果和平方误差
    clusterAssment=mat(zeros((m,2)))
    #质心初始化为所有数据点的均值
    centroid0=mean(dataSet,axis=0).tolist()[0]
    #初始化只有一个质心的list
    centList=[centroid0]
    #计算所有数据点到初始质心的距离平方误差
    for j in range(m):
        clusterAssment[j,1]=distMeas(mat(centroid0),dataSet[j,:])**2
    while len(centList)<k:
        lowestSSE=inf
        #对于每个质心
        for i in range(len(centList)):
            #获取当前族i下的所有数据点
            ptsInCurrCluster=dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #将当前族i进行二分kMeans处理
            centroidMat,splitClustAss=kMeans(ptsInCurrCluster,2,distMeas)
            #将二分kMeans结果中的平方和距离求和
            sseSplit=sum(splitClustAss[:,1])
            #将未参与二分k-Means分配结果中的平方和的距离求和
            sseNotSplit=sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if sseSplit+sseNotSplit<lowestSSE:
                bestCentToSplit=i
                bestNewCents=centroidMat
                bestClustAss=splitClustAss.copy()
                lowestSSE=sseSplit+sseNotSplit
        #找出最好的族分配结果，默认族是0,1
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit
        print("the bestCentToSplit is: ",bestCentToSplit)
        print("the len of bestClustAss is: ",len(bestClustAss))
        #更新质心列表
        #更新质心list中的第i个质心为使用二分kMeans后bestNewCents的第一个质心
        centList[bestCentToSplit]=bestNewCents[0,:].tolist()[0]
        #添加第二个质心
        centList.append(bestNewCents[1,:].tolist()[0])
        #重新分配最好族下的数据以及sse
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    return mat(centList),clusterAssment


    
if __name__=='__main__':
    # datMat=mat(loadDataSet('testSet.txt'))
    # myCentroids,clustAssing=kMeans(datMat,4)

    dataMat=mat(loadDataSet('testSet2.txt'))
    centList,myNewAssments=biKmeans(dataMat,3)
    print(centList)
    