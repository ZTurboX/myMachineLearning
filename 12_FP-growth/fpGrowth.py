class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name=nameValue#节点名称
        self.count=numOccur #节点出现的次数
        self.nodeLink=None   #链接相似的元素
        self.parent=parentNode #指向父节点
        self.children={} #存储叶子节点

    def inc(self,numOccur):
        self.count+=numOccur
    
    def disp(self,ind=1):
        '''
        树以文本形式显示
        '''
        print(' '*ind,self.name,' ',self.count)
        for child in self.children.values():
            child.disp(ind+1)


def updateHeader(nodeToTest,targetNode):
    '''
    更新头指针，建立相同元素间的关系
    nodeToTest:满足minSup的字典
    targetNode:Tree对象的子节点
    '''
    while nodeToTest.nodeLink is not None:
        nodeToTest=nodeToTest.nodeLink
    nodeToTest.nodeLink=targetNode


def updateTree(items,inTree,headerTable,count):
    '''
    items:满足minSup排序后的元素key的数组
    inTree:空Tree对象
    headerTable:满足minSup的字典
    count:原数据中每一组key出现的次数
    '''
    #取出元素出现次数最高的
    #如果该元素在inTree.children字典中，就累加
    #如果该元素不存在就新增key,value为初始化的treeNode对象
    if items[0] in inTree.children:
        #更新最大元素，对应的treeNode对象的count进行叠加
        inTree.children[items[0]].inc(count)
    else:
        #如果不存在子节点，为该inTree添加子节点
        inTree.children[items[0]]=treeNode(items[0],count,inTree)
        #如果满足minSup的字典的value第二位为null,就设置该元素为本节点对应的tree节点
        #如果元素第二位不为null,就更新header节点
        if headerTable[items[0]][1] is None:
            #headerTable记录第一次节点出现的位置
            headerTable[items[0]][1]=inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items)>1:
        updateTree(items[1:],inTree.children[items[0]],headerTable,count)



def createTree(dataSet,minSup=1):
    '''
    创建Fp-tree:
    1.遍历数据集获得每个元素项的出现频率
    2.去掉不满足最小支持度的元素项
    3.构建FP树：
        读入每个项集并将其添加到一条已存在的路径中：
            如果该路径不存在，则创建一条新路径

    minSup:最小的支持度
    '''
    #支持度>=minSup的字典
    headerTable={}
    for trans in dataSet:
        #统计每一行中每个元素出现的总次数
        for item in trans:
            headerTable[item]=headerTable.get(item,0)+dataSet[trans]
    #删除headerTable中，元素次数<最小支持度的元素
    for k in list(headerTable.keys()):
        if headerTable[k]<minSup:
            del(headerTable[k])
    #满足minSup元素集合
    freqItemSet=set(headerTable.keys())
    if len(freqItemSet)==0:
        return None,None
    for k in headerTable:
        #格式化:dist{key:[元素次数,None]}
        headerTable[k]=[headerTable[k],None]
    #创建树
    retTree=treeNode('Null Set',1,None)
    for tranSet,count in dataSet.items():
        localD={}
        #根据频率对每个事物中的元素排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item]=headerTable[item][0]
        if len(localD)>0:
            #取出元组的key值
            orderedItems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree,headerTable


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict={}
    for trans in dataSet:
        retDict[frozenset(trans)]=1
    return retDict

def ascendTree(leafNode,prefixPath):
    '''
    如果父节点存在，就记录当前节点的name值

    leafNode:要查询的节点所在当前nodeTree
    prefixPath:前缀路径
    '''
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

def findPrefixPath(basePat,treeNode):
    '''
    basePat:要查询的节点值
    treeNode:查询的节点所在的当前nodeTree
    '''
    #对非basePat的倒序值作为key,赋值为count
    conPats={}
    while treeNode is not None:
        prefixPath=[]
        #寻找该节点的父节点
        ascendTree(treeNode,prefixPath)
        if len(prefixPath)>1:
            conPats[frozenset(prefixPath[1:])]=treeNode.count
        treeNode=treeNode.nodeLink
    return conPats

def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    '''
    创建条件FP树
    headerTable:满足minSup所有的元素字典
    minSup:最小支持项集
    freqItemList:存储频繁子项的列表
    '''
    #从小到大排序，得到频繁项集的key
    bigL=[v[0] for v in sorted(headerTable.items(),key=lambda p:p[1][0])]
    for basePat in bigL:
        newFreqSet=preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases=findPrefixPath(basePat,headerTable[basePat][1])
        myCondTree,myHead=createTree(condPattBases,minSup)
        if myHead is not None:
            myCondTree.disp(1)
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)





if __name__=='__main__':
    simpDat=loadSimpDat()
    initSet=createInitSet(simpDat)
    myFPtree,myHeaderTab=createTree(initSet,3)
    #myFPtree.disp()
    freqItems=[]
    mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)
    print(mineTree)