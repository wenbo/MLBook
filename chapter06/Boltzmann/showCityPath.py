# -*- coding:utf-8 -*-
# Filename : testBoltzmann01.py

import operator
import copy
import Untils
from numpy import *
import matplotlib.pyplot as plt
# 加载数据文件
def loadDataSet(fileName): 
	numFeat = len(open(fileName).readline().split('\t')) - 1 
	dataMat = []; 
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		lineArr.append(float(curLine[0]));
		lineArr.append(float(curLine[1]));
		dataMat.append(lineArr)
	return dataMat

# 绘制二维数据集坐标散点图:无分类
# 适用于 List 和 Matrix
def drawScatter(dataMat,plt):
		px = (dataMat[:,0]).tolist()
		py = (dataMat[:,1]).tolist()	
		plt.scatter(px,py,c='green',marker='o',s=60)
		i=65
		for x,y in zip(px,py):
			plt.annotate(str(chr(i))+"("+str(int(x[0]))+","+str(int(y[0]))+")",xy =(x[0]+40,y[0]),color='black',fontsize=10)	
			i += 1
		
def drawPath(Seq,dataMat,plt,color='b'):
		m,n = shape(dataMat)	
		px = (dataMat[Seq,0]).tolist()
		py = (dataMat[Seq,1]).tolist()
		px.append(px[0]); py.append(py[0])
		plt.plot(px,py,color) 

dataSet = loadDataSet("dataSet25.txt")
cityPosition = mat(dataSet)
m,n = shape(cityPosition)


# 优化前城市图,路径图
drawScatter(cityPosition,plt)
'''
drawPath(range(m),cityPosition,plt) 
'''
plt.show()