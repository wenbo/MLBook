# -*- coding: utf-8  -*-

from numpy import *
import sys
import matplotlib.pyplot as plt 

def scatterplot(k,wMat):# 绘制图形
	fig = plt.figure();	ax = fig.add_subplot(111) 
	m,n=shape(wMat)
	for i in xrange(m): #逐列描点
		ax.scatter(mat(k),wMat[:,i],s=0.1,marker=".")
	plt.show()	
	
maxIter =1000 # 最大迭代数和系数分辨率区间
k= linspace(2.1,4.0,maxIter) # logisitic区间
klen = len(k)
xMat =mat(zeros((klen,maxIter)))  # 初始化结果矩阵
x = 1.0/float(maxIter)
for i in xrange(klen) :   # 沿系数方向循环
	for j in xrange(maxIter):  
		x = float(k[i])*x*(1.0-x) # 变量迭代
		xMat[i,j]=x
# 绘制图形
scatterplot(k,xMat)	