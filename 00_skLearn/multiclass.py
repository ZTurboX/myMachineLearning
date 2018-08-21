#多类和多标签算法
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

#1对1，多类别学习
# iris=datasets.load_iris()
# X,y=iris.data,iris.target
# res=OneVsOneClassifier(LinearSVC(random_state=0)).fit(X,y).predict(X)
# print(res)


#误差校正输出代码，多类别学习
# iris=datasets.load_iris()
# X,y=iris.data,iris.target
# res=OutputCodeClassifier(LinearSVC(random_state=0),code_size=2,random_state=0).fit(X,y).predict(X)
# print(res)


#多输出回归
# from sklearn.datasets import make_regression
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# X,y=make_regression(n_samples=10,n_targets=3,random_state=1)
# res=MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X,y).predict(X)
# print(res)


#多输出分类
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np 
X,y1=make_classification(n_samples=10,n_features=100,n_informative=30,n_classes=3,random_state=1)
y2=shuffle(y1,random_state=1)
y3=shuffle(y1,random_state=2)
Y=np.vstack((y1,y2,y3)).T
n_samples,n_features=X.shape
n_outputs=Y.shape[1]
n_classes=3
forest=RandomForestClassifier(n_estimators=100,random_state=1)
multi_target_forest=MultiOutputClassifier(forest,n_jobs=-1)
res=multi_target_forest.fit(X,Y).predict(X)
print(res)