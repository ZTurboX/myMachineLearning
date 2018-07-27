# 基于概率论的分类方法:朴素贝叶斯

## 贝叶斯理论

假设p1(x,y)表示数据点(x,y)属于类别1的概率，p2(x,y)表示数据点(x,y)属于类别2的概率，对于一个新数据点，可用下面规则判断类别：

若p1(x,y)>p2(x,y),类别为1

若p2(x,y)>p1(x,y),类别为2

贝叶斯决策理论的核心思想就是选择高概率对应的类别，即选择具有最高概率的决策

贝叶斯准则：

![贝叶斯准则](https://github.com/TonyJent/myMachineLearning/blob/master/images/04_NaiveBayes/%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%87%86%E5%88%99.PNG)

其中P(c)是类“先验”概率，P(x|c)是样本x相对于类标记c的类条件概率，P(x)是用于归一化的“证据”因子。

## 朴素贝叶斯原理

朴素贝叶斯常用于文档分类，通常有两种实现方式：伯努利模型，基于多项式模型，我们在这里使用伯努利模型。

基于属性条件独立性假设，贝叶斯准则又可写成：

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/04_NaiveBayes/%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%87%86%E5%88%992.PNG)

### 工作原理

提取所有文档中的词条并去重

获取文档的所有类别

计算每个类别中的文档数目

对每篇文档：

​    对每个类别：

​        如果词条出现在文档中-->增加该词条的计数值

​        增加所有词条的计数值（此类别下词条总数）

对每个类别：

​    对每个词条：

​        将该词条的数目除以总词条数目得到的条件概率(P(词条|类别))

返回该文档属于每个类别的条件概率(P(类别|文档的所有词条))

## 核心代码

朴素贝叶斯训练函数

```python
def trainNB0(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类训练函数：
    trainMatrix:文件单词矩阵
    trainCategory:文件对应的类别
    '''
    #文件数
    numTrainDocs=len(trainMatrix)
    #单词数
    numWords=len(trainMatrix[0])
    #计算侮辱性文件的出现的概率，trainCategory所有1的求和/文件总数
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #构造单词出现次数列表
    #p0Num=zeros(numWords)
    #p1Num=zeros(numWords)
    #整个数据集单词出现总数
    #p0Denom=0.0
    #p1Denom=0.0
    '''
    利用朴素贝叶斯对文档分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，
    即计算p(w0|1)p(w1|1)p(w2|1)
    如果其中一个概率为0，那么最后乘积也为0.所以将词出现数初始化为1，分母初始化为2
    '''
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
   
    for i in range(numTrainDocs):
        #是否是侮辱性文件
        if trainCategory[i]==1:
            #如果是侮辱性文件，对侮辱性文件的向量相加
            p1Num+=trainMatrix[i] 
            #对向量中的所有元素求和，计算所有侮辱性文件中出现的单词总数
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    
    #在类别1下每个单词出现的概率
    #p1Vect=p1Num/p1Denom
    #在类别0下每个单词出现的概率
    #p0Vect=p0Num/p0Denom

    '''
    由于太多很小的数相乘，会造成下溢，则将p1Vect,p0Vect分别取对数
    '''
    #在类别1下每个单词出现的概率
    p1Vect=log(p1Num/p1Denom)
    #在类别0下每个单词出现的概率
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
```

朴素贝叶斯分类函数

```python
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''
    朴素贝叶斯分类函数:将乘法转换为加法
    乘法:P(C|F1F2....Fn)=P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
    加法:P(F1|C)*P(F2|C)...P(Fn|C)P(C)-->log(P(F1|C))+log(P(F2|c))....+log(P(Fn|C))+log(P(C))
    
    vec2Classify:待测数据，要分类的向量
    p0Vec:类别0，[log(P(F1|C0)),log(P(F2|C0))....log(P(Fn|C0))]列表
    p1Vec:类别1，[log(P(F1|C1)),log(P(F1|C1))....log(P(F1|C1))]列表
    pClass1:类别1，侮辱性文件出现的概率
    '''
    #计算log(P(F1|C))+log(P(F2|c))....+log(P(Fn|C))+log(P(C))
    p1=sum(vec2Classify*p1Vec)+log(pClass1)#P(w|c1)*P(c1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)#P(w|c0)*P(c0)
    if p1>p0:
        return 1
    else:
        return 0
```

[朴素贝叶斯进行文档分类完整代码](https://github.com/TonyJent/myMachineLearning/blob/master/04_NaiveBayes/bayes.py)

## 项目实战

[使用朴素贝叶斯过滤垃圾邮件](https://github.com/TonyJent/myMachineLearning/blob/master/04_NaiveBayes/filterSpam.py)

[使用朴素贝叶斯分类器从个人广告中获取区域倾向](https://github.com/TonyJent/myMachineLearning/blob/master/04_NaiveBayes/personalAdvertising.py)

