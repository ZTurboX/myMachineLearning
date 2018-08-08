from numpy import *
from numpy import linalg as la
import SVD


def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k])>thresh:
                print(1)
            else:
                print(0)
        print('')

def imgCompress(numSV=3,thresh=0.8):
    '''
    图像压缩
    numSV:Sigma长度
    '''
    myl=[]
    for line in open('0_5.txt').readlines():
        newRow=[]
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)

    myMat=mat(myl)
    print("*****************original matrix******************")
    printMat(myMat,thresh)
    U,Sigma,VT=la.svd(myMat)
    SigRecon=mat(zeros((numSV,numSV)))
    #SigRecon=mat(eye(numSV)*Sigma[:numSV])
    for k in range(numSV):
        SigRecon[k,k]=Sigma[k]
    reconMat=U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****************reconstructed matrix using %d singular values*************************" % numSV)
    printMat(reconMat,thresh)

if __name__=='__main__':
    imgCompress(2)