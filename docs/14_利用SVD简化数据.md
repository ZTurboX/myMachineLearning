# 利用SVD简化数据

## 矩阵分解

![](https://github.com/TonyJent/myMachineLearning/blob/master/images/14_SVD/%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3.PNG)

## 推荐系统

### 相似度计算

1. 欧式距离法

   ```python
   def euclidSim(inA,inB):
       '''
       欧式距离法
       '''
       return 1.0/(1.0+la.norm(inA-inB))
   ```

2. 皮尔逊相关系数

   ```python
   def pearsSim(inA,inB):
       '''
       皮尔逊相关系数
       '''
       if len(inA)<3:
           return 1.0
       return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]
   ```

3. 余弦相似度

   ```python
   def cosSim(inA,inB):
       '''
       余弦相似度
       '''
       num=float(inA.T*inB)
       denom=la.norm(inA)*la.norm(inB)
       return 0.5+0.5*(num/denom)
   ```

## 项目实战

[餐馆菜肴推荐引擎](https://github.com/TonyJent/myMachineLearning/blob/master/14_SVD/foodRecommend.py)

[基于SVD的图像压缩](https://github.com/TonyJent/myMachineLearning/blob/master/14_SVD/imgsCompress.py)



