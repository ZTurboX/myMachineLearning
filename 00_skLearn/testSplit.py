
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

#交叉验证
iris=datasets.load_iris()
# print(iris.data.shape,iris.target.shape)
# X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.4,random_state=0)
# print(X_train.shape,y_train.shape)
# print(X_test.shape,y_test.shape)
# clf=svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
# print(clf.score(X_test,y_test))


#计算交叉验证的指标
from sklearn.model_selection import cross_val_score
clf=svm.SVC(kernel='linear',C=1)
# scores=cross_val_score(clf,iris.data,iris.target,cv=5)
# print(scores)
# print("accuracy:%0.2f (+/- %0.2f)"%(scores.mean(),scores.std()*2))
# scores=cross_val_score(clf,iris.data,iris.target,cv=5,scoring='f1_macro')
# print(scores)
#交叉验证迭代器
# from sklearn.model_selection import ShuffleSplit
# n_samples=iris.data.shape[0]
# cv=ShuffleSplit(n_splits=3,test_size=0.3,random_state=0)
# scores=cross_val_score(clf,iris.data,iris.target,cv=cv)
# print(scores)


#多度量评估
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
# # scoring=['precision_macro','recall_macro']
# # clf=svm.SVC(kernel='linear',C=1,random_state=0)
# # scores=cross_validate(clf,iris.data,iris.target,scoring=scoring,cv=5,return_train_score=False)
# # print(sorted(scores.keys()))
# # print(scores['test_recall_macro'])

from sklearn.metrics.scorer import make_scorer
# scoring={'prec_macro':'precision_macro','rec_micro':make_scorer(recall_score,average='macro')}
# scores=cross_validate(clf,iris.data,iris.target,scoring=scoring,cv=5,return_train_score=True)
# print(sorted(scores.keys()))
# print(scores['train_rec_micro'])



#通过交叉验证获取预测
from sklearn.model_selection import cross_val_predict
# predicted=cross_val_predict(clf,iris.data,iris.target,cv=10)
# res=metrics.accuracy_score(iris.target,predicted)
# print(res)


#交叉验证迭代器
#k折
import numpy as np 
from sklearn.model_selection import KFold
# X=["a","b","c","d"]
# kf=KFold(n_splits=2)
# for train,test in kf.split(X):
#     print("%s %s" % (train,test))


#重复k折交叉验证
from sklearn.model_selection import RepeatedKFold
# X=np.array([[1,2],[3,4],[1,2],[3,4]])
# random_state=12883823
# rkf=RepeatedKFold(n_splits=2,n_repeats=2,random_state=random_state)
# for train,test in rkf.split(X):
#     print("%s %s" % (train,test))


#留一交叉验证
# from sklearn.model_selection import LeaveOneOut
# X=[1,2,3,4]
# loo=LeaveOneOut()
# for train,test in loo.split(X):
#     print("%s %s" % (train,test))

#留p交叉验证
# from sklearn.model_selection import LeavePOut
# X=np.ones(4)
# lpo=LeavePOut(p=2)
# for train,test in lpo.split(X):
#     print("%s %s" % (train,test))



#随机排列交叉验证
# from sklearn.model_selection import ShuffleSplit
# X=np.arange(5)
# ss=ShuffleSplit(n_splits=3,test_size=0.25,random_state=0)
# for train_index,test_index in ss.split(X):
#     print("%s %s" % (train_index,test_index))



#基于类标签，具有分层的交叉验证迭代器

#分层k折
# from sklearn.model_selection import StratifiedKFold
# X=np.ones(10)
# y=[0,0,0,0,1,1,1,1,1,1]
# skf=StratifiedKFold(n_splits=3)
# for train,test in skf.split(X,y):
#     print("%s %s" % (train,test))


#用于分组数据的交叉验证迭代器
# from sklearn.model_selection import GroupKFold
# X=[0.1,0.2,2.2,2.4,2.3,4.55,5.8,8.8,9,10]
# y=["a","b","b","b","c","c","c","d","d","d"]
# groups=[1,1,1,2,2,2,3,3,3,3]
# gkf=GroupKFold(n_splits=3)
# for train,test in gkf.split(X,y,groups=groups):
#     print("%s %s" % (train,test))


#留一组交叉验证
# from sklearn.model_selection import LeaveOneGroupOut
# X=[1,5,10,50,60,70,80]
# y=[0,1,1,2,2,2,2]
# groups=[1,1,2,2,3,3,3]
# logo=LeaveOneGroupOut()
# for train,test in logo.split(X,y,groups=groups):
#     print("%s %s" % (train,test))


#留p组交叉验证
from sklearn.model_selection import LeavePGroupsOut
X=np.arange(6)
y=[1,1,1,2,2,2]
groups=[1,1,2,2,3,3]
lpgo=LeavePGroupsOut(n_groups=2)
for train,test in lpgo.split(X,y,groups=groups):
    print("%s %s" % (train,test))