#随机梯度下降
import numpy as np 
from sklearn import linear_model

#分类

# X=np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
# Y=np.array([1,1,2,2])
# clf=linear_model.SGDClassifier()
# clf.fit(X,Y)
# print(clf.predict([[-0.8,-1]]))
# #模型参数
# print(clf.coef_)
# #截距
# print(clf.intercept_)
# #超平面的符号距离
# print(clf.decision_function([[2.,2.]]))


#回归
n_samples,n_features=10,5
np.random.seed(0)
y=np.random.randn(n_samples)
X=np.random.randn(n_samples,n_features)
clf=linear_model.SGDRegressor()
clf.fit(X,y)

