from sklearn import svm



#分类
# X=[[0,0],[1,1]]
# y=[0,1]
# clf=svm.SVC()
# clf.fit(X,y)
# res_predicted=clf.predict([[2.,2.]])
# #获得支持向量
# res_support_vector=clf.support_vectors_
# #获得支持向量的索引
# res_support=clf.support_
# #每一个类别获得支持向量的数量
# res_nSupport=clf.n_support_
# print(res_nSupport)


#多元分类
# X=[[0],[1],[2],[3]]
# Y=[0,1,2,3]
# clf=svm.SVC(decision_function_shape='ovo')
# clf.fit(X,Y)
# dec=clf.decision_function([[1]])
# print(dec.shape[1])


#回归
from sklearn import svm
X=[[0,0],[2,2]]
y=[0.5,2.5]
clf=svm.SVR()
clf.fit(X,y)
res=clf.predict([[1,1]])
print(res)
