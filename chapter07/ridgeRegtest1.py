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
def scatterplot(wMat,logk):# 绘制图形
	fig = plt.figure()
	ax = fig.add_subplot(111) 
	wMatT = wMat.T
	m,n=shape(wMatT)
	for i in xrange(m):
		ax.plot(logk,wMatT[i,:])
		ax.annotate("feature["+str(i)+"]",xy =(i,wMatT[i,0]),color='black')	
	plt.show()	

			
# 前8列为xArr,后1列为yArr
xArr,yArr = loadDataSet("ridgedata2.txt")
xMat,yMat= normData(xArr,yArr) # 标准化数据集

Knum = 100 # 确定lam的范围exp(-10~20)
# 初始化30行,8列的全0矩阵
wMat = zeros((Knum,shape(xMat)[1]))
klist = zeros((Knum,1))
for i in xrange(Knum):
	k = i/1000.0   # 算法的目的是确定k的取值
	klist[i]=k
	xTx = xMat.T*xMat
	denom = xTx + eye(shape(xMat)[1])*k
	if linalg.det(denom) == 0.0:
		print "This matrix is singular, cannot do inverse"
		sys.exit(0) 
	ws = denom.I * (xMat.T*yMat)
	wMat[i,:]=ws.T
print klist
scatterplot(klist,klist) # k值的变化
scatterplot(wMat,klist)  # 岭回归

