# -*- coding: utf-8  -*-

from numpy import *
import sys
import os
import matplotlib.pyplot as plt 

def loadDataSet(fileName): 
    numFeat = len(open(fileName).readline().split('\t')) - 1  
    X= []; Y = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')          
        X.append([float(curLine[i]) for i in xrange(numFeat) ])
        Y.append(float(curLine[-1]))
    return X,Y
#绘制图形
def plotscatter(Xmat,Ymat,yHat,plt):
	fig = plt.figure()
	ax = fig.add_subplot(111)  # 绘制图形位置
	ax.scatter(Xmat,Ymat,c='blue',marker='o')	# 绘制散点图 
	plt.plot(Xmat,yHat,'r')	# 绘制散点图
	plt.show()

# 数据矩阵,分类标签
xArr,yArr = loadDataSet("nolinear.txt")
# 局部加权线性回归算法：回归线矩阵

# RBF函数的平滑系数
miu= 0.02
k = 0.03

# 数据集坐标数组转换为矩阵
xMat = mat(xArr); yMat = mat(yArr).T
testArr = xArr # 测试数组
m,n = shape(xArr) # xArr的行数
yHat = zeros(m) # yHat是y的预测值,yHat的数据是y的回归线矩阵
for i in xrange(m):
    weights = mat(eye(m))
    for j in xrange(m):                          
        diffMat = testArr[i] - xMat[j,:] 
        # 利用高斯核函数计算权重矩阵,计算后的权重是一个对角阵	
        weights[j,j] = exp(diffMat*diffMat.T/(-miu*k**2)) 
    xTx = xMat.T * (weights * xMat) # 矩阵左乘自身的转置
    if linalg.det(xTx) != 0.0:   
       ws = xTx.I * (xMat.T * (weights * yMat))
       yHat[i] = testArr[i] * ws # 计算回归线坐标矩阵
    else: 
       print "This matrix is singular, cannot do inverse"
       sys.exit(0)  # 退出程序

plotscatter(xMat[:,1],yMat,yHat,plt) # 绘制图形

# 计算相关系数:
print corrcoef(yHat,yMat.T)

