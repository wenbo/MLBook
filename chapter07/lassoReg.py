# -*- coding: utf-8  -*-
# Filename : 05stepWise.py

from numpy import *
import sys
import matplotlib.pyplot as plt 

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat	
# 矩阵标准化
def normData(xArr,yArr):
	xMat = mat(xArr); yMat=mat(yArr).T
	yMean = mean(yMat,0) 
	xMeans = mean(xMat,0) 
	ynorm = yMat - yMean
	xVar = var(xMat,0)
	xnorm = (xMat - xMeans)/xVar
	return xnorm,ynorm
def scatterplot(wMat,k):# 绘制图形
	fig = plt.figure()
	ax = fig.add_subplot(111) 
	wMatT = wMat.T
	m,n=shape(wMatT)
	for i in xrange(m):
		ax.plot(k,wMatT[i,:])
		ax.annotate("feature["+str(i)+"]",xy =(0,wMatT[i,0]),color='black')	
	plt.show()	
	
# 前8列为xArr,后1列为yArr
xArr,yArr = loadDataSet("ridgedata2.txt")
# 数据矩阵转换
xMat,yMat=normData(xArr,yArr)
m,n = shape(xMat)
eps = 0.005 # 迭代步长变化
numIt = 1000 # 迭代次数

returnMat = zeros((numIt,n)) #返回矩阵
ws = zeros((n,1)) # 初始化ws为全零向量
wsTest = ws.copy(); wsMax = ws.copy()
for i in xrange(numIt):
    lowestError = inf; # 初始化lowestError为无穷大
    for j in xrange(n): # n 为特征向量的维度
        for sign in [-1,1]: # sign:信号量 取值为-1和1
            wsTest = ws.copy()
            wsTest[j] += eps*sign # 信号量乘以步进值 
            yTest = xMat*wsTest # xMat乘以wsTest为特征向量
            rssE = ((yMat.A-yTest.A)**2).sum() # 误差计算公式 # .A返回自身数据的一个引用(不进行拷贝)
            if rssE < lowestError: # 判别最小误差
                lowestError = rssE # 更新最小误差值
                wsMax = wsTest # 更新wsMax
    ws = wsMax.copy()
    returnMat[i,:] = ws.T
print returnMat

# 绘制图形
# lasso
fig = plt.figure()
ax = fig.add_subplot(111) 
ax.plot(returnMat)
plt.show()