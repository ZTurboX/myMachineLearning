from numpy import *
import random

def loadDataSet(filename):
    dataset=[]
    with open(filename,'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            lineArr=[]
            for feature in line.split(','):
                #返回移除字符串头尾指定的字符生成的新字符串
                strF=feature.strip()
                if strF.isdigit():
                    lineArr.append(float(strF))
                else:
                    lineArr.append(strF)
            dataset.append(lineArr)
    return dataset

def cross_validation_split(dataset,n_folds):
    '''
    样本数据随机无放回抽样
    dataset:原始数据集
    n_folds:数据集分成n_folds份
    '''
    dataset_split=list()
    dataset_copy=list(dataset)
    fold_size=len(dataset)/n_folds
    for i in range(n_folds):
        fold=list()
        while len(fold)<fold_size:
            index=random.randrange(len(dataset_copy))
            #无放回抽样
            fold.append(dataset_copy[index])
        dataset_split.append(fold)
    return dataset_split

def test_split(index,value,dataset):
    '''
    根据特征和特征值分割数据集
    index:特征的下标
    value:要比较的特征值
    '''
    left,right=list(),list()
    for row in dataset:
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)
    return left,right


def gini_index(groups,class_values):
    '''
    计算代价
    groups=(left,right)
    class_values=[0,1]
    '''
    gini=0.0
    for class_value in class_values:
        for group in groups:
            size=len(group)
            if size==0:
                continue
            proportion=[row[-1] for row in group].count(class_value)/float(size)
            gini+=(proportion*(1.0-proportion))
    return gini

def get_split(dataset,n_features):
    '''
    找出分割数据集的最优特征，得到最优特征index,特征值row[index],分割完数据groups(left,right)
    '''
    #数据集dataset中分类标签的集合
    class_values=list(set(row[-1] for row in dataset))
    b_index,b_value,b_score,b_groups=999,999,999,None
    features=list()
    while len(features)<n_features:
        #随机选取特征的index
        index=random.randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            #遍历每一行index索引下的特征值作为分类值value,找出最优的分类特征和特征值
            groups=test_split(index,row[index],dataset)
            gini=gini_index(groups,class_values)
            if gini<b_score:
                b_index,b_value,b_score,b_groups=index,row[index],gini,groups
    return {'index':b_index,'value':b_value,'groups':b_groups}


def to_treminal(group):
    '''
    输出group中出现次数较多的标签
    '''
    outcomes=[row[-1] for row in group]
    return max(set(outcomes),key=outcomes.count)

def split(node,max_depth,min_size,n_features,depth):
    '''
    创建子分割器，递归分类，直到分类结束
    '''
    left,right=node['groups']
    del(node['groups'])
    #check for a no split
    if not left or not right:
        node['left']=node['right']=to_treminal(left+right)
        return 
    #check for max depth
    if depth>=max_depth:
        node['left'],node['right']=to_treminal(left),to_treminal(right)
        return 
    #process left child
    if len(left)<=min_size:
        node['left']=to_treminal(left)
    else:
        node['left']=get_split(left,n_features)
        split(node['left'],max_depth,min_size,n_features,depth+1)
    #process right child
    if len(right)<=min_size:
        node['right']=to_treminal(right)
    else:
        node['right']=get_split(right,n_features)
        split(node['right'],max_depth,min_size,n_features,depth+1)

def build_tree(train,max_depth,min_size,n_features):
    '''
    创建一个决策树
    train:训练数据
    min_size:叶子结点的大小
    n_features:选取特征的个数
    '''

    root=get_split(train,n_features)
    split(root,max_depth,min_size,n_features,1)
    return root

def predict(node,row):
    '''
    预测模型分类结果
    '''
    if row[node['index']]<node['value']:
        if isinstance(node['left'],dict):
            return predict(node['left'],row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
        else:
            return node['right']


def bagging_predict(trees,row):
    '''
    make a prediction with a list of bagged trees
    trees:决策树的集合
    row:测试数据集的每一行数据
    '''
    predictions=[predict(tree,row) for tree in trees]
    return max(set(predictions),key=predictions.count)
    


def subsample(dataset,ratio):
    '''
    训练数据随机化
    ratio:训练数据集的样本比例
    '''
    #随机抽样的训练样本
    sample=list()
    #训练样本按比例抽样，返回浮点数x的四舍五入值
    n_sample=round(len(dataset)*ratio)
    while len(sample)<n_sample:
        #有放回的随机抽样
        index=random.randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def random_forest(train,test,max_depth,min_size,sample_size,n_trees,n_features):
    '''
    随机森林算法：评估算法性能，返回模型得分

    train:训练数据集
    test:测试数据集
    min_size:叶子结点大小
    sample_size:训练数据集样本比例
    n_trees:决策树个数
    n_features:选取特征的个数
    '''

    trees=list()
    for i in range(n_trees):
        #随机抽样的训练样本
        sample=subsample(train,sample_size)
        tree=build_tree(sample,max_depth,min_size,n_features)
        trees.append(tree)

    predictions=[bagging_predict(trees,row) for row in test]
    return predictions

def accuracy_metric(actual,predicted):
    '''
    计算精确度
    '''
    correct=0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            correct+=1
    return correct/float(len(actual))*100.0

def evaluate_algorithm(dataset,algorithm,n_folds,*args):
    '''
    评估算法性能，返回模型得分
    algorithm:使用的算法
    '''
    #将数据集进行抽样n_folds份
    folds=cross_validation_split(dataset,n_folds)
    scores=list()
    for fold in folds:
        train_set=list(folds)
        train_set.remove(fold)
        train_set=sum(train_set,[])
        test_set=list()
        for row in fold:
            row_copy=list(row)
            row_copy[-1]=None
            test_set.append(row_copy)
        predicted=algorithm(train_set,test_set,*args)
        actual=[row[-1] for row in fold]

        accuracy=accuracy_metric(actual,predicted)
        scores.append(accuracy)
    return scores
    

if __name__=='__main__':
    dataset=loadDataSet('sonar-all-data.txt')
    n_folds=5
    max_depth=20
    min_size=1
    sample_size=1.0
    n_features=15
    for n_trees in [1,10,20]:
        scores=evaluate_algorithm(dataset,random_forest,n_folds,max_depth,min_size,sample_size,n_trees,n_features)
        seed(1)
        print('random=',random())
        print('Trees: %d' % n_trees)
        print('scores: %s' % scores)
        print('Mean accuracy: %.3f%%' % (sum(scores)/float(scores)))