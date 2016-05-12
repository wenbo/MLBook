# -*- coding: utf-8 -*-
import os
import sys
import numpy as np 
import operator
from numpy import *
from common_libs import *
import matplotlib.pyplot as plt 


Input = file2matrix("testSet.txt","\t")
target = Input[:,-1] #获取分类标签列表
[m,n] = shape(Input) 

dataMat = buildMat(Input)

# 4. 定义迭代次数 
steps = 500  # 迭代次数
weights = ones(n) # 初始化权重向量

alphalist =[]
alphahlist =[]
# 算法主程序:
# 1.对数据集的每个行向量进行m次随机抽取
# 2.对抽取之后的行向量应用动态步长
# 3.进行梯度计算
# 4.求得行向量的权值，合并为矩阵的权值
for j in xrange(steps):
	dataIndex = range(m) # 以导入数据的行数m为个数产生索引向量:0~99
	for i in xrange(m):
		alpha = 2/(1.0+j+i)+0.0001  #动态修改alpha步长从4->0.016
		if j==0: alphalist.append(alpha)
		if i==0: alphahlist.append(alpha)
		randIndex = int(random.uniform(0,len(dataIndex)))	#生成0~m之间随机索引
		vectSum = sum(dataMat[randIndex]*weights.T) # 计算dataMat随机索引与权重的点积和
		grad = logistic(vectSum) # 计算点积和的梯度
		errors = target[randIndex]-grad # 计算误差
		weights = weights + alpha * errors * dataMat[randIndex] #计算行向量权重
		del(dataIndex[randIndex]) #从数据集中删除选取的随机索引	

# print weights	# 输出训练后的权重
weights	= weights.tolist()[0]
lenal=  len(alphalist); lenalh=  len(alphahlist)
fig = plt.figure()
axes1 = plt.subplot(211); axes2 = plt.subplot(212)
X1 = np.linspace(0,lenal,lenal); X2 = np.linspace(0,lenalh,lenalh)
axes1.plot(X1,alphalist); axes2.plot(X2,alphahlist)
plt.show()
