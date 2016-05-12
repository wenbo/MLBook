# -*- coding: utf-8 -*-
# Filename : 02kMeans1.py

from numpy import *
import numpy as np
from Recommand_Lib import *
import matplotlib.pyplot as plt 

# 从文件构建的数据集    
dataMat = file2matrix("testData/4k2_far.txt","\t")  
dataSet = mat(dataMat[:,1:]) # 转换为矩阵形式
 

k = 4 # 分类数
m = shape(dataSet)[0]
# 初始化第一个聚类中心: 每一列的均值
centroid0 = mean(dataSet, axis=0).tolist()[0] 
centList =[centroid0] # 把均值聚类中心加入中心表中
# 初始化聚类距离表,距离方差: 
ClustDist = mat(zeros((m,2)))
for j in range(m):
	ClustDist[j,1] = distEclud(centroid0,dataSet[j,:])**2 	

# 依次生成k个聚类中心
while (len(centList) < k):
    lowestSSE = inf # 初始化最小误差平方和。核心参数，这个值越小就说明聚类的效果越好。
    # 遍历cenList的每个向量
    #----1. 使用ClustDist计算lowestSSE，以此确定:bestCentToSplit、bestNewCents、bestClustAss----#
    for i in xrange(len(centList)):
        ptsInCurrCluster = dataSet[nonzero(ClustDist[:,0].A==i)[0],:]
        # 应用标准kMeans算法(k=2),将ptsInCurrCluster划分出两个聚类中心,以及对应的聚类距离表	
        centroidMat,splitClustAss = kMeans(ptsInCurrCluster, 2)
        # 计算splitClustAss的距离平方和
        sseSplit = sum(splitClustAss[:,1])
        # 计算ClustDist[ClustDist第1列!=i的距离平方和
        sseNotSplit = sum(ClustDist[nonzero(ClustDist[:,0].A!=i)[0],1])
        if (sseSplit + sseNotSplit) < lowestSSE: # 算法公式: lowestSSE = sseSplit + sseNotSplit
            bestCentToSplit = i                 # 确定聚类中心的最优分隔点 
            bestNewCents = centroidMat          # 用新的聚类中心更新最优聚类中心
            bestClustAss = splitClustAss.copy() # 深拷贝聚类距离表为最优聚类距离表
            lowestSSE = sseSplit + sseNotSplit  # 更新lowestSSE       
    # 回到外循环
    #----2. 计算新的ClustDist----#
    # 计算bestClustAss 分了两部分:
    # 第一部分为bestClustAss[bIndx0,0]赋值为聚类中心的索引     
    bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
    # 第二部分 用最优分隔点的指定聚类中心索引  
    bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
    # 以上为计算bestClustAss
    # 更新ClustDist对应最优分隔点下标的距离，使距离值等于最优聚类距离对应的值
    #以上为计算ClustDist
    
    #----3. 用最优分隔点来重构聚类中心----#
    # 覆盖: bestNewCents[0,:].tolist()[0]附加到原有聚类中心的bestCentToSplit位置
    # 增加: 聚类中心增加一个新的bestNewCents[1,:].tolist()[0]向量
    centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
    centList.append(bestNewCents[1,:].tolist()[0]) 
    ClustDist[nonzero(ClustDist[:,0].A == bestCentToSplit)[0],:]= bestClustAss 
    # 以上为计算centList
color_cluster(ClustDist[:,0:1],dataSet,plt)
print "cenList:",mat(centList)
# print "ClustDist:", ClustDist
# 绘制聚类中心图形
drawScatter(plt,mat(centList),size=60,color='red',mrkr='D')

plt.show()

