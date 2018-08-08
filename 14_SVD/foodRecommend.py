from numpy import *
from numpy import linalg as la
import SVD


def standEst(dataMat,user,simMeas,item):
    '''
    计算在给定相似度计算方法的条件下，用户对物品的估计评分值
    dataMat:训练数据集
    user:用户编号
    simMeas:相似度计算方法
    item:未评分的物品编号
    '''
    #物品数目
    n=shape(dataMat)[1]
    #初始化两个评分值
    simTotal=0.0
    ratSimTotal=0.0
    for j in range(n):
        userRating=dataMat[user,j]
        #如果某个物品评分值为0，则跳过这个物品
        if userRating==0: 
            continue
        #寻找两个用户都评分的物品
        #overLap返回对菜item和j都评过分的用户id
        #logical_and计算两个元素的真值
        overLap=nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        print(overLap)
        #如果没有两个重合元素，相似度为0
        if len(overLap)==0:
            similarity=0
        else:
            #如果存在重合的物品，则基于这些重合物品计算相似度
            #similarity为用户相似度
            similarity=simMeas(dataMat[overLap,item],dataMat[overLap,j])
        simTotal+=similarity
        #userRating为用户评分
        ratSimTotal+=similarity*userRating
        if simTotal==0:
            return 0
        else:
            #归一化
            return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=SVD.cosSim,estMethod=standEst):
    #寻找未评级物品
    unratedItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0:
        return 'you rated everything'
    itemScores=[]
    for item in unratedItems:
        #寻找前N个未评级物品，预测得分
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

def svdEst(dataMat,user,simMeas,item):
    '''
    基于SVD的评分估计
    '''
    n=shape(dataMat)[1]
    simTotal=0.0
    ratSimTotal=0.0
    #奇异值分解，只利用包含90%能量值的奇异值
    U,Sigma,VT=la.svd(dataMat)
    #对奇异值构建对角矩阵
    Sig4=mat(eye(4)*Sigma[:4])
    #利用U矩阵将物品转换到低维空间
    xformedItems=dataMat.T*U[:,:4]*Sig4.I
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0 or j==item:
            continue
        similarity=simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print("the %d and %d similarity is: %f" % (item,j,similarity))
        simTotal+=similarity
        ratSimTotal+=similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal



if __name__=='__main__':
    myMat=mat(SVD.loadExData())
    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
    myMat[3,3]=2
    print(recommend(myMat,1,estMethod=svdEst,simMeas=SVD.pearsSim)) 