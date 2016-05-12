# -*- coding: utf-8  -*-
# Filename : 02regression.py

from numpy import *
import sys
import os
import matplotlib.pyplot as plt 

def loadDataSet(fileName): 
    X = []; Y = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        X.append(float(curLine[0])); Y.append(float(curLine[-1]))
    return X,Y
#绘制图形
def plotscatter(Xmat,Ymat,a,b,plt):
	fig = plt.figure()
	ax = fig.add_subplot(111)  # 绘制图形位置
	ax.scatter(Xmat,Ymat,c='blue',marker='o')	# 绘制散点图
	Xmat.sort() # 对Xmat各元素进行排序	
	yhat = [a*float(xi)+b for xi in Xmat] # 计算预测值	
	plt.plot(Xmat,yhat,'r') # 绘制回归线		
	plt.show()
	return yhat

# 数据矩阵,分类标签
xArr,yArr = loadDataSet("regdataset.txt")
# 生成X坐标列
m = len(xArr)
Xmat = mat(ones((m,2)))
for i in xrange(m): Xmat[i,1] = xArr[i] 
Ymat = mat(yArr).T # 转换为y列

xTx = Xmat.T*Xmat  # 矩阵左乘自身的转置

ws = []
if linalg.det(xTx) != 0.0:
    # 计算直线的斜率和截距
    # 矩阵正规方程组公式:inv(X.T*X)*X.T*Y	
    ws = xTx.I * (Xmat.T*Ymat)
else: 
    print "This matrix is singular, cannot do inverse"
    sys.exit(0)  # 退出程序
print "ws:",ws
yHat = plotscatter(Xmat[:,1],Ymat,ws[1,0],ws[0,0],plt)

# 计算相关系数:
print corrcoef(yHat,Ymat.T)
