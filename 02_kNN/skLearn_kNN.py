import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import neighbors

#找到最近邻
# X=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
# nb=NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(X)
# distances,indices=nb.kneighbors(X)
# print(indices)
# print(distances)


#kNN分类
# group=[[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]
# labels=['A','A','B','B']
# X=[[0],[1],[2],[3]]
# y=[0,0,1,1]
# nb=KNeighborsClassifier(n_neighbors=3)
# nb.fit(group,labels)
# print(nb.predict([[0,0]]))


n_neighbors=15
iris=datasets.load_iris()
#print(iris.data[:50])
X=iris.data[:,:2]
y=iris.target
#网格中的步长
h=.02
cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])
for weights in ['uniform','distance']:
    clf=neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X,y)
    #画出决策边界
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    #meshgrid从坐标向量返回坐标矩阵
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),
                      np.arange(y_min,y_max,h))
    #np.c_按行连接两个矩阵，把两矩阵左右相加，要求行数相等
    #ravel():将多维数组降为一维
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])

    Z=Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)

    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title("3-class classification (k=%i,weights='%s')" % (n_neighbors,weights))
plt.show()

