# -*- coding: GBK -*-
# Filename :gradDecent.py

from numpy import *
import operator
import Untils
import matplotlib.pyplot as plt 

# BP神经网络

# 数据集: 列1:截距 1;列2:x坐标; 列3:y坐标
dataMat,classLabels = Untils.loadDataSet("student.txt")
dataMat = mat(dataMat)
classMat= mat(classLabels)

# 数据归一化
dataMat = Untils.normalize(dataMat)

# 绘制数据集坐标散点图
Untils.drawClassScatter(dataMat,classLabels,False)
		
# m行数 n列数
m,n = shape(dataMat)
labelMat = classMat.transpose()
# 步长
alpha = 0.001
# 迭代次数
maxCycles = 500
#构成线性分割线 y=a*x+b: b:weights[0]; a:weights[1]/weights[2]
weights = ones((n,1))
# 计算回归系数 weights
for k in range(maxCycles):
	# 通过sigmoid函数返回结果，h是梯度计算的结果，是一个列向量
	# h = logRegres2.sigmoid(dataMatrix*weights)
	h = 1.0/(1+exp(-dataMat*weights))
	# 误差计算:分类标签(0,1)-h
	error = (labelMat - h)  
	# 回归系数：更新权重
	weights = weights + alpha * dataMat.transpose()* error 
print weights	

# 绘制分类线图形
Untils.ClassifyLine(-3,3,weights)
	
