from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

#随机森林
# X=[[0,0],[1,1]]
# y=[0,1]
# clf=RandomForestClassifier(n_estimators=10)
# clf=clf.fit(X,y)
# print(clf)

#极限随机树
# X,y=make_blobs(n_samples=10000,n_features=10,centers=100,random_state=0)

# # clf=DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)

# #clf=RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)

# clf=ExtraTreesClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)

# scores=cross_val_score(clf,X,y)
# print(scores.mean())


#AdaBoost
# iris=load_iris()
# clf=AdaBoostClassifier(n_estimators=100)
# scores=cross_val_score(clf,iris.data,iris.target)
# print(scores.mean())


#梯度树提升
#分类
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

# X,y=make_hastie_10_2(random_state=0)
# X_train,X_test=X[:2000],X[2000:]
# y_train,y_test=y[:2000],y[2000:]
# clf=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0).fit(X_train,y_train)
# print(clf.score(X_test,y_test))

#回归
import numpy as np 
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

X,y=make_friedman1(n_samples=1200,random_state=0,noise=1.0)
X_train,X_test=X[:200],X[200:]
y_train,y_test=y[:200],y[200:]
est=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=1,random_state=0,loss='ls').fit(X_train,y_train)
res=mean_squared_error(y_test,est.predict(X_test))
print(res)