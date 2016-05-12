# -*- coding: GBK -*-
# Filename : 03BPTest.py

from numpy import *
import operator
import Untils
import BackPropgation
import matplotlib.pyplot as plt 

# 数据集
dataSet,classLabels = BackPropgation.loadDataSet("testSet2.txt") # 初始化时第1列为全1向量, studentTest.txt
dataSet = BackPropgation.normalize(mat(dataSet))

# 绘制数据点
# 重构dataSet数据集
dataMat = mat(ones((shape(dataSet)[0],shape(dataSet)[1])))
dataMat[:,1] = mat(dataSet)[:,0]
dataMat[:,2] = mat(dataSet)[:,1]	

# 绘制数据集散点图
Untils.drawClassScatter(dataMat,transpose(classLabels),False)

# BP神经网络进行数据分类
errRec,WEX,wex = BackPropgation.bpNet(dataSet,classLabels)

# 计算和绘制分类线
x,z = BackPropgation.BPClassfier(-3.0,3.0,WEX,wex)

Untils.classfyContour(x,x,z)

# 绘制误差曲线
X = linspace(0,2000,2000)
Y = log2(errRec)+1.0e-6
Untils.TrendLine(X,Y)
