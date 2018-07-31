from numpy import *
import adaboost
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def plotROC(predStrengths,classLabels):
    '''
    predStrengths:分类器最终预测结果的权重值
    '''
    #绘制光标的位置
    cur=(1.0,1.0)
    #计算AUC的值
    ySum=0.0
    #对正样本求和
    numPosClas=sum(array(classLabels)==1.0)
    #正样本概率
    yStep=1/float(numPosClas)
    xStep=1/float(len(classLabels)-numPosClas)
    #从大到小排序的索引值
    sortedIndicies=predStrengths.argsort()
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        #标签为1的类，沿y轴方向下降一个步长，不断降低真阳率
        if classLabels[index]==1.0:
            delX=0
            delY=yStep
        #对于其他的，在x方向倒退一个步长
        else:
            delX=xStep
            delY=0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=(cur[0]-delX,cur[1]-delY)
        #对角线虚线
        ax.plot([0,1],[0,1],'b--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve for AdaBoost horse colic detection system')
        ax.axis([0,1,0,1])
        plt.show()
        print("the area under the curve is: ",ySum*xStep)



if __name__=='__main__':
    datArr,labelArr=loadDataSet('horseColicTraining2.txt')
    classifierArray,aggClassEst=adaboost.adaBoostTrainDS(datArr,labelArr,10)
    plotROC(aggClassEst.T,labelArr)
    #print(classifierArray)
    #testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
    #prediction10=adaboost.adaClassify(testArr,classifierArray)
    #print(prediction10)
    #errArr=mat(ones((67,1)))
    #print(errArr[prediction10!=mat(testLabelArr).T].sum())