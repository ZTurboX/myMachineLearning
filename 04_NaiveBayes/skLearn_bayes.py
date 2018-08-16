import numpy as np 
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

#伯努利朴素贝叶斯
# X=np.random.randint(2,size=(6,100))
# Y=np.array([1,2,3,4,4,5])
# clf=BernoulliNB()
# clf.fit(X,Y)
# print(clf.predict(X[2:3]))


#高斯朴素贝叶斯
X=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
Y=np.array([1,1,1,2,2,2])
clf=GaussianNB()
clf.fit(X,Y)
print(clf.predict([[-0.8,-1]]))
