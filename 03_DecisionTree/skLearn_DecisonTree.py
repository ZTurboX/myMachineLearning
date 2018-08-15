from sklearn import tree
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 

#Decison Tree Classify
# X=[[0,0],[1,1]]
# Y=[0,1]
# clf=tree.DecisionTreeClassifier()
# clf=clf.fit(X,Y)
# print(clf.predict([[2.,2.]]))



n_classes=3
plot_colors="bry"
plot_step=0.02

iris=load_iris()

for pairidx,pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
    X=iris.data[:,pair]
    y=iris.target

    clf=DecisionTreeClassifier().fit(X,y)

    plt.subplot(2,3,pairidx+1)

    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,plot_step),
                      np.arange(y_min,y_max,plot_step))
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    cs=plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    for i,color in zip(range(n_classes),plot_colors):
        idx=np.where(y==i)
        plt.scatter(X[idx,0],X[idx,1],c=color,label=iris.target_names[i],cmap=plt.cm.Paired)
    plt.axis("tight")

plt.suptitle("decision surface of a decision tree using paired features")
plt.legend()
plt.show()