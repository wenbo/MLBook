# -*- coding: utf-8 -*-

import sys  
import os
from numpy import *
# 设置utf-8 unicode环境
reload(sys)
sys.setdefaultencoding('utf-8')

def splitDataSet(dataSet, axis, value):		
		retDataSet = []     
		for featVec in dataSet:
			if featVec[axis] == value:
				reducedFeatVec = featVec[:axis] # list操作 提取0~(axis-1)的元素 
				reducedFeatVec.extend(featVec[axis+1:]) # list操作 将特征轴（列）之后的元素加回
				retDataSet.append(reducedFeatVec) 
		# 返回划分后的特征矩阵
		return retDataSet 
# P1=128.0/384.0
# P2=256.0/384.0
P1=257.0/384.0
P2=127.0/384.0
Ip1p2=-P1*log2(P1)-P2*log2(P2)
print Ip1p2

mylist = [1,0,1,0,1,0,0]
items = dict([(mylist.count(i), i) for i in mylist])
print items[max(items.keys())]
dataset = [[1,0],[0,1],[1,0]]
numEntries = len(dataset) # 得到数据集行数  
labelCounts = {}          # 初始化类别标签	for featVec in dataset: # 这段代码计算了数据集中各个特征向量的和
for featVec in dataset :
    currentLabel = featVec[-1]
    if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
    labelCounts[currentLabel] += 1
print labelCounts 
cateList = [data[-1] for data in dataset] # 从数据集中得到类别标签
items = dict([(cateList.count(i), i) for i in cateList]) # 得到类别为key，出现次数value的字典          
print items

print splitDataSet(dataset, 0, 0)	  

P1=640.0/1024.0
P2=384.0/1024.0
Ip1p2=-P1*log2(P1)-P2*log2(P2)
print Ip1p2
print 0.9544-0.6877 