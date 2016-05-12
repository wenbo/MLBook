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

# 3.构建b+x 系数矩阵：b这里默认为1
dataMat = buildMat(Input)
# print dataMat
# 4. 定义步长和迭代次数 
alpha = 0.001 # 步长
steps = 500  # 迭代次数
weights = ones((n,1))# 初始化权重向量
weightlist = []
# 5. 主程序
for k in xrange(steps):
	gradient = dataMat*mat(weights) # 梯度
	output = logistic(gradient)  # logistic函数
	errors = target-output # 计算误差
	weights = weights + alpha*dataMat.T*errors 
	weightlist.append(weights) 

print weights	# 输出训练后的权重
fig = plt.figure()
axes1 = plt.subplot(311)
axes2 = plt.subplot(312)
axes3 = plt.subplot(313)
weightmat = mat(zeros((steps,n)))
i=0
for weight in weightlist:
	weightmat[i,:]=weight.T
	i+= 1
X =linspace(0,steps,steps)
# 输出前10个点的截距变化
axes1.plot(X,weightmat[:,0],color = 'blue', linewidth=1, linestyle="-") 
axes1.set_ylabel('weight[0]')	
axes2.plot(X,weightmat[:,1],color = 'red', linewidth=1, linestyle="-") 	
axes2.set_ylabel('weight[1]')		
axes3.plot(X,weightmat[:,2],color = 'green', linewidth=1, linestyle="-") 	
axes3.set_ylabel('weight[2]')	
plt.show()
