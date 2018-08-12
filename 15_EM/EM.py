from numpy import *

def initData(k,u,sigma,dataNum):
    '''
    初始化高斯混合模型的数据
    k:比例系数
    u:均值
    sigma:标准差
    dataNum:数据个数
    '''
    dataSet=zeros(dataNum,dtype=float)
    #高斯分布个数
    n=len(k)
    for i in range(dataNum):
        #产生0-1的随机数
        rand=random.random()
        sK=0
        index=0
        while index<n:
            sK+=k[index]
            if rand<sK:
                dataSet[i]=random.normal(u[index],sigma[index])
                break
            else:
                index+=1
    return dataSet


def normalFun(x,u,sigma):
    '''
    计算均值为u,标准差为sigma的正太分布函数的密度函数值
    '''
    return (1.0/sqrt(2*pi)*sigma)*(exp(-(x-u)**2/(2*sigma**2)))

def em(dataSet,k,u,sigma,step=10):
    '''
    高斯混合模型
    '''
    n=len(k)
    dataNum=len(dataArr)
    gamaArr=zeros((n,dataNum))
    for s in range(step):
        #E步，确定Q函数
        for i in range(n):
            for j in range(dataNum):
                wSum=sum([k[t]*normalFun(dataSet[j],u[t],sigma[t]) for t in range(n)])
                gamaArr[i][j]=k[i]*normalFun(dataSet[j],u[i],sigma[i])/float(wSum)
        
        #M步
        #更新u
        for i in range(n):
            u[i]=sum(gamaArr[i]*dataSet)/sum(gamaArr[i])
        #更新sigma
        for i in range(n):
            sigma[i]=sqrt(sum(gamaArr[i]*(dataSet-u[i])**2)/sum(gamaArr[i]))
        #更新k
        for i in range(n):
            k[i]=sum(gamaArr[i])/dataNum
    
    return [k,u,sigma]

if __name__=='__main__':
    #参数的准确值
    k=[0.3,0.4,0.3]
    u=[2,4,3]
    sigma=[1,1,4]
    #样本数
    dataNum=5000
    dataArr=initData(k,u,sigma,dataNum)

    k0=[0.3,0.3,0.4]
    u0=[1,2,2]
    sigma0=[1,1,1]
    step=100

    k1,u1,sigma1=em(dataArr,k0,u0,sigma0,step)
    print("参数实际值：")
    print("k:",k)
    print("u:",u)
    print("sigma:",sigma)

    print("参数估计值：")
    print("k1:",k1)
    print("u1:",u1)
    print("sigma1:",sigma1)


