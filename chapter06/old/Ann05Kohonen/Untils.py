# -*- coding: utf-8 -*-
# Filename : Untils.py
'''
Created on Oct 27, 2010
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
    plt.scatter(px,py,c='blue',marker='o')
    if flag : displayplot();

# 路径
def drawPath(Seq,dataMat,color='b',flag=True):
    px = (dataMat[Seq,0]).tolist()[0]
    py = (dataMat[Seq,1]).tolist()[0]
    px.append(px[0]); py.append(py[0])
    plt.plot(px,py,color) 
    if flag : displayplot();
    	    
# 绘制二维数据集坐标散点图:有分类
# 适用于 List 和 Matrix
def drawClassScatter(dataMat,classLabels,flag=True):
    # 绘制list
    if type(dataMat) is list :
    	i = 0
    	for mydata in dataMat:
    		if classLabels[i]==0:
    			plt.scatter(mydata[1],mydata[2],c='blue',marker='o')
    		else:
    			plt.scatter(mydata[1],mydata[2],c='red',marker='s')	
    		i +=1;
    # 绘制Matrix	
    if type(dataMat) is matrix :
    	i = 0
    	for mydata in dataMat:
    		if classLabels[i]==0:
    			plt.scatter(mydata[0,1],mydata[0,2],c='blue',marker='o')
    		else:
    			plt.scatter(mydata[0,1],mydata[0,2],c='red',marker='s')	
    		i +=1;    	    
    if flag : displayplot();

# 绘制分类线
def ClassifyLine(begin,end,weights,flag=True):
	# 确定初始值和终止值,精度	
	X = linspace(begin,end,(end-begin)*100)
	# 建立线性分类方差
	Y = -(float(weights[0])+float(weights[1])*X)/float(weights[2]) 
	plt.plot(X,Y,'b')
	if flag : displayplot()

# 绘制趋势线: 可调整颜色		
def TrendLine(X,Y,color='r',flag=True):
	plt.plot(X,Y,color)
	if flag : displayplot()
		
# 合并两个多维的Matrix，并返回合并后的Matrix
# 输入参数有先后顺序    
def mergMatrix(matrix1,matrix2):
    [m1,n1] = shape(matrix1)
    [m2,n2] = shape(matrix2)
    if m1 != m2:
    	print "different rows,can not merge matrix"
    	return; 	
    mergMat = zeros((m1,n1+n2))
    mergMat[:,0:n1] = matrix1[:,0:n1]
    mergMat[:,n1:(n1+n2)] = matrix2[:,0:n2]
    return mergMat 	

# 计算相关系数
# 入口: x,y 的元素个数必须相同的一维数组
# start 开始计算相关的下标 >=0
# 返回: 相关系数
def corref( x , y, start = 0 ):
    N = len(x)
    if (N!=len(y)) or (N<start+2):
       return 0.0
    Sxx=Syy=Sxy=Sx=Sy=0
    for i in range(start,N):
        Sx = Sx + x[i]
        Sy = Sy + y[i]
    Sx = Sx / (N - start)
    Sy = Sy / (N - start)
    for i in range(start,N):
        Sxx = Sxx + (x[i]- Sx )*(x[i] - Sx)
        Syy = Syy + (y[i]- Sy )*(y[i] - Sy)
        Sxy = Sxy + (x[i]- Sx )*(y[i] - Sy)
    r = abs( Sxy ) / math.sqrt(Sxx * Syy )
    return r	
