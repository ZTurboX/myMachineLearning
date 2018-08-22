#特征选择

from sklearn.feature_selection import VarianceThreshold

#移除低方差特征
# X=[[0,0,1],[0,1,0],[1,0,0],[0,1,1],[0,1,0],[0,1,1]]
# sel=VarianceThreshold(threshold=(.8*(1-.8)))
# res=sel.fit_transform(X)
# print(res)



#单变量特征选择
#SelectKBest移除那些除了评分最高的K个特征之外的所有特征
#SelectPercentile移除除了用户指定的最高得分百分比之外的所有特征
#对每个特征应用常见的单变量统计测试：假阳性率SelectFpr,伪发现率SelectFdr,族系误差SelectFwe
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# iris=load_iris()
# X,y=iris.data,iris.target
# print(X.shape)
# X_new=SelectKBest(chi2,k=2).fit_transform(X,y)
# print(X_new.shape)


#使用SelectFromModel选取特征

# #基于L1的特征选取
# from sklearn.svm import LinearSVC
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectFromModel
# iris=load_iris()
# X,y=iris.data,iris.target
# print(X.shape)
# lsvc=LinearSVC(C=0.01,penalty="l1",dual=False).fit(X,y)
# model=SelectFromModel(lsvc,prefit=True)
# X_new=model.transform(X)
# print(X_new.shape)

#基于Tree的特征选取
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris=load_iris()
X,y=iris.data,iris.target
print(X.shape)
clf=ExtraTreesClassifier()
clf=clf.fit(X,y)
print(clf.feature_importances_)
model=SelectFromModel(clf,prefit=True)
X_new=model.transform(X)
print(X_new.shape)
