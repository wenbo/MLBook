# -*- coding: utf-8 -*-
import os
import sys
import numpy as np 
import operator
from numpy import *
from common_libs import *
import matplotlib.pyplot as plt 

# 1.导入数据
Input = file2matrix("testSet.txt","\t")
target = Input[:,-1] #获取分类标签列表
[m,n] = shape(Input) 

# 2.按分类绘制散点图
drawScatterbyLabel(plt,Input)

# 3.构建b+x 系数矩阵：b这里默认为1
dataMat = buildMat(Input)
# print dataMat
# 4. 定义步长和迭代次数 
steps = 500  # 迭代次数
weights = ones(n) # 初始化权重向量

# 算法主程序:
# 1.对数据集的每个行向量进行m次随机抽取
# 2.对抽取之后的行向量应用动态步长
# 3.进行梯度计算
# 4.求得行向量的权值，合并为矩阵的权值
for j in xrange(steps):
	dataIndex = range(m) # 以导入数据的行数m为个数产生索引向量:0~99
	for i in xrange(m):
		alpha = 2/(1.0+j+i)+0.0001  #动态修改alpha步长从4->0.016
		randIndex = int(random.uniform(0,len(dataIndex)))	#生成0~m之间随机索引
		vectSum = sum(dataMat[randIndex]*weights.T) # 计算dataMat随机索引与权重的点积和
		grad = logistic(vectSum) # 计算点积和的梯度
		errors = target[randIndex]-grad # 计算误差
		weights = weights + alpha * errors * dataMat[randIndex] #计算行向量权重
		del(dataIndex[randIndex]) #从数据集中删除选取的随机索引	

print weights	# 输出训练后的权重
weights	= weights.tolist()[0]
# 6. 绘制训练后超平面
X = np.linspace(-5,5,100)
#y=w*x+b: b:weights[0]/weights[2]; w:weights[1]/weights[2]
Y = -(double(weights[0])+X*(double(weights[1])))/double(weights[2])
plt.plot(X,Y)
plt.show()
