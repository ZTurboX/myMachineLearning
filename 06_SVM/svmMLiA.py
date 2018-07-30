'''
SMO思想:将大优化问题分解为多个小优化问题求解
SMO原理:每次循环选择两个alpha进行优化处理，一旦找出一对合适的alpha,那么就增大一个同时减少一个
        alpha必须要在间隔边界之外
        alpha还没进行过区间化处理或不在边界上
'''
import random
from numpy import *
def loadDataSet(fileName):
    '''
    对文件解析
    '''
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    '''
    i:第一个alpha下标
    m:所有alpha数目
    '''
    j=i
    #只要函数值不等于输入值i，函数就会随机选择
    while j==i:
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    '''
    用于调整大于H或小于L的alpha值
    '''
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

'''
SMO算法：
创建一个alpha向量并将其初始化为0向量
当迭代次数小于最大迭代次数时：
    对数据集中的每个数据向量：
        如果该数据向量可以被优化：
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
    如果所有向量都没被优化，增加迭代数目，继续下一次循环
'''

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    '''
    dataMatLn:特征集合
    classLabels:类别标签
    C:松弛常量，允许有些数据点可以处于分割面的错误一侧
                控制最大化间隔，保证大部分的函数间隔小于1.0这两个目标权重
    toler:容错率：某个体系中能减小一些因素或选择对某个系统产生不稳定的概率
    maxIter:退出最大的循环次数

    return:
        b:模型的常量值
        alphas:拉格朗日乘子
    '''
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)

    b=0
    alphas=mat(zeros((m,1)))
    #没有任何alpha改变的情况下遍历数据的次数
    iter=0
    while iter<maxIter:
        #记录alpha是否已经优化，每次循环设为0，然后再对整个集合顺序遍历
        alphaPairsChanged=0
        for i in range(m):
            #f=w^T*x[i]+b=Σa[n]*label[n]*x[n]*x[i]+b
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            #预测结果与真实结果的误差
            Ei=fXi-float(labelMat[i])
        
            '''
            检验样本(xi,yi)是否满足KKT条件：
            yi*f(i) >= 1 and alpha=0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha<C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the  boundary)

            发生错误的概率labelMat[i]*Ei如果超出了toler,才需要优化
            '''
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                #如果满足优化条件，随机选取非i的点，进行优化比较
                j=selectJrand(i,m)
                #预测j的结果
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()

                #L,H用于将alphas[j]调整到0-C之间。如果L==H就不做改变
                #labelMat[i]!=labelMat[j]表示异侧，就相减，否则是同侧，相加
                if labelMat[i]!=labelMat[j]:
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                #如果相同，就不优化
                if L==H:
                    print("L==H")
                    continue

                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print("eta>=0")
                    continue
                
                #计算新的alphas[j]值
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                #用L和H对其调整
                alphas[j]=clipAlpha(alphas[j],H,L)
                #检查alphas[j]是否是轻微的改变，如果是退出循环
                if abs(alphas[j]-alphaJold)<0.00001:
                    print("j not moving enough")
                    continue
                #对alphas[i]同样进行改变
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

                #计算阈值b
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T

                if 0<alphas[i] and C>alphas[i]:
                    b=b1
                elif 0<alphas[i] and C>alphas[j]:
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if alphaPairsChanged==0:
            iter+=1
        else:
            iter=0
        print("iteration number: %d" % iter)
    return b,alphas

if __name__=='__main__':
    dataArr,labelArr=loadDataSet('testSet.txt')
    b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
    shape(alphas[alphas>0])
    for i in range(100):
        if alphas[i]>0.0:
            print(dataArr[i],labelArr[i])





