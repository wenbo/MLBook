# -*- coding: utf-8  -*-
# Filename : 04ridgeTest.py

from numpy import *
import sys
import matplotlib.pyplot as plt 
# 岭回归函数
# 加载数据集
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
def normData(xMat,yMat):
	xMat = mat(xArr); yMat=mat(yArr).T
	yMean = mean(yMat,0) 
	xMeans = mean(xMat,0) 
	ynorm = yMat - yMean
	xVar = var(xMat,0)
	xnorm = (xMat - xMeans)/xVar
	return xnorm,ynorm
def scatterplot(wMat):# 绘制图形
	fig = plt.figure()
	ax = fig.add_subplot(111) 
	wMatT = wMat.T
	m,n=shape(wMatT)
	for i in xrange(m):
		ax.plot(wMatT[i,:])
		ax.annotate("feature["+str(i)+"]",xy =(i,wMatT[i,0]),color='black')	
	plt.show()	
def Multicollinear(xMat):
	features = xMat.T
	m,n = shape(features) 
	for i in xrange(m):		
		if i==(m-1):
			print i ,":", 0 
			print corrcoef(features[i],features[0])	
		else:
			print i ,":", i+1 			
			print corrcoef(features[i],features[i+1])
			
# 前8列为xArr,后1列为yArr
xArr,yArr = loadDataSet("ridgedata.txt")
xMat,yMat= normData(xArr,yArr) # 标准化数据集
Multicollinear(xMat)



