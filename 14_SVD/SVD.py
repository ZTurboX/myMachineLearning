from numpy import *
from numpy import linalg as la

def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

# Data=loadExData()
# U,Sigma,VT=linalg.svd(Data)
# print(Sigma)


'''
相似度计算
'''
def euclidSim(inA,inB):
    '''
    欧式距离法
    '''
    return 1.0/(1.0+la.norm(inA-inB))

def pearsSim(inA,inB):
    '''
    皮尔逊相关系数
    '''
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    '''
    余弦相似度
    '''
    num=float(inA.T*inB)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

if __name__=='__main__':
    myMat=mat(loadExData())
    res=euclidSim(myMat[:,0],myMat[:,4])
    print(res)
