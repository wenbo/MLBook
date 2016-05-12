# -*- coding: utf-8 -*-
# Filename : Recommand_lib.py

from numpy import *
import numpy as np 
import operator
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt 

def savefile(savepath,content):
	fp = open(savepath,"wb")
	fp.write(content)
	fp.close()

# 数据文件转矩阵
# path: 数据文件路径
# delimiter: 文件分隔符
def file2matrix(path,delimiter):	
	recordlist = []
	fp = open(path,"rb") 	# 读取文件内容
	content = fp.read()
	fp.close()
	rowlist = content.splitlines() 	# 按行转换为一维表
	# 逐行遍历 		# 结果按分隔符分割为行向量
	recordlist=[map(eval, row.split(delimiter)) for row in rowlist if row.strip()]	
	return mat(recordlist)	# 返回转换后的矩阵形式

# 欧氏距离
eps = 1.0e-6
def distEclud(vecA, vecB):
	return linalg.norm(vecA-vecB)+eps 
# 相关系数
def distCorrcoef(vecA, vecB):
	return corrcoef(vecA, vecB, rowvar = 0)[0][1]
# Jaccard距离	
def distJaccard(vecA, vecB):
	temp = mat([array(vecA.tolist()[0]),array(vecB.tolist()[0])])
	return dist.pdist(temp,'jaccard')
# 余弦相似度
def cosSim(vecA, vecB):	
	return (dot(vecA,vecB.T)/((linalg.norm(vecA)*linalg.norm(vecB))+eps))[0,0]
# 绘制散点图    
def drawScatter(plt,mydata,size=20,color='blue',mrkr='o'):
	m,n = shape(mydata)
	if m>n and m>2:
		plt.scatter(mydata.T[0],mydata.T[1],s=size,c=color,marker=mrkr)   
	else:
		plt.scatter(mydata[0],mydata[1],s=size,c=color,marker=mrkr)   

# 绘制分类点
def drawScatterbyLabel(plt,Input):
	m,n=shape(Input)
	target = Input[:,-1] 
	for i in xrange(m):
		if target[i]==1:
			plt.scatter(Input[i,0],Input[i,1],c='blue',marker='o')
		else:
			plt.scatter(Input[i,0],Input[i,1],c='red',marker='s')	


# 硬限幅函数
def hardlim(dataSet):
	dataSet[nonzero(dataSet.A>0)[0]]=1
	dataSet[nonzero(dataSet.A<=0)[0]]=0	
	return dataSet

# Logistic函数
def logistic(wTx):
	return 1.0/(1.0+exp(-wTx))

def buildMat(dataSet):
	m,n=shape(dataSet)
	dataMat = zeros((m,n))
	dataMat[:,0] = 1 
	dataMat[:,1:] = dataSet[:,:-1]
	return 	dataMat
	
# 分类函数
def classifier(testData, weights):
    prob = logistic(sum(testData*weights)) # 求取概率--判别算法
    if prob > 0.5: 
    	return 1.0 # prob>0.5 返回为1
    else: 
    	return 0.0          # prob<=0.5 返回为0	

# 最小二乘回归，用于测试    	
def standRegres(xArr,yArr):
	xMat = mat(ones((len(xArr),2)))
	yMat = mat(ones((len(yArr),1)))
	xMat[:,1:] = (mat(xArr).T)[:,0:]	
	yMat[:,0:] = (mat(yArr).T)[:,0:]
	xTx = xMat.T*xMat
	if linalg.det(xTx) == 0.0:
		print "This matrix is singular, cannot do inverse"
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws