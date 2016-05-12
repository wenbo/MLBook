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
print dataMat[:10,:]
# 4. 定义步长和迭代次数 
alpha = 0.001 # 步长
steps = 500  # 迭代次数
weights = ones((n,1))# 初始化权重向量
# 5. 主程序
for k in xrange(steps):
	gradient = dataMat*mat(weights) # 梯度
	output = logistic(gradient)  # logistic函数
	errors = target-output # 计算误差
	weights = weights + alpha*dataMat.T*errors  

print weights	# 输出训练后的权重
# 6. 绘制训练后超平面
X = np.linspace(-7,7,100)
#y=w*x+b: b:weights[0]/weights[2]; w:weights[1]/weights[2]
Y = -(double(weights[0])+X*(double(weights[1])))/double(weights[2])
plt.plot(X,Y)
plt.show()
