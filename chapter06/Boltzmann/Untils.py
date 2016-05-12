# -*- coding: utf-8 -*-
# Filename : Untils.py
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: jack zheng
'''
from numpy import *
import operator
import matplotlib.pyplot as plt 

# 加载数据文件
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; 
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        lineArr.append(float(curLine[0]));
        lineArr.append(float(curLine[1]));
        dataMat.append(lineArr)
    return dataMat

# 显示绘制图形
def displayplot():
    plt.show()	
    
# 绘制二维数据集坐标散点图:无分类
# 适用于 List 和 Matrix
def drawScatter(dataMat,flag=True):
    if type(dataMat) is list :
    	px = (mat(dataMat)[:,0]).tolist()
    	py = (mat(dataMat)[:,1]).tolist()	
    if type(dataMat) is matrix :
    	px = (dataMat[:,0]).tolist()
    	py = (dataMat[:,1]).tolist()	
    plt.scatter(px,py,c='green',marker='o',s=60)
    i=65
    for x,y in zip(px,py):
      plt.annotate(str(chr(i)),xy =(x[0]+40,y[0]),color='red')	
      i += 1	
    if flag : displayplot();

# 路径
def drawPath(Seq,dataMat,color='b',flag=True):
    m,n = shape(dataMat)	
    px = (dataMat[Seq,0]).tolist()
    py = (dataMat[Seq,1]).tolist()
    px.append(px[0]); py.append(py[0])
    plt.plot(px,py,color) 
    if flag : displayplot();

# 绘制趋势线: 可调整颜色		
def TrendLine(X,Y,color='r',flag=True):
	plt.plot(X,Y,color)
	if flag : displayplot()
		