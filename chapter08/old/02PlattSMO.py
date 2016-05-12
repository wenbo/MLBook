# -*- coding: utf-8 -*-
# Filename : 02PlattSMO.py

from numpy import *
import numpy as np
import operator
import svmMLiA2
import matplotlib.pyplot as plt 

dataArr, labelArr = svmMLiA2.loadDataSet('nolinear.txt')
# print labelArr
# 主 platt smo 函数
# 数据集:dataArr 
# 类别标签:labelArr, 
# 错分类系数C: 0.6, 
# 容错率:0.001
# 迭代次数: 40
b, alphas = svmMLiA2.smoP(dataArr, labelArr, 0.6, 0.001, 200)
# 根据拉格朗日alphas乘子计算W向量
ws = svmMLiA2.calcWs(alphas, dataArr, labelArr)

print "b:",b
print "alphas[alphas > 0]:",alphas[alphas > 0]

# 绘制散点图
mydata = mat(dataArr)
# 数据描点
fig = plt.figure()
ax = fig.add_subplot(111) 
for i in range(len(mydata)):
	if alphas[i]!=0: # KKT条件
		ax.scatter(mydata[i,0],mydata[i,1],c='green',marker='s')		
	elif labelArr[i] == 1:
		ax.scatter(mydata[i,0],mydata[i,1],c='blue',marker='o')
	elif labelArr[i] == -1:
		ax.scatter(mydata[i,0],mydata[i,1],c='red',marker='o')
# 显示绘制的图形
plt.show()