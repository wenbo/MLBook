# -*- coding: GBK -*-
# Filename : 01dataSet.py

import numpy as np 
import operator
import Untils
import BackPropgation
from numpy import *
import matplotlib.pyplot as plt 

dataMat,classLabels = Untils.loadDataSet("student.txt")

# 绘制图形：二维散点,无分类
# Untils.drawScatter(dataMat)

# 绘制图形：二维散点,有分类,适合训练集
# Untils.drawClassScatter(mat(dataMat),classLabels)

# 合并两个多维的matrix，并返回合并后的Matrix
# 输入参数有先后顺序
# [m,n]=shape(dataMat)
# classMat = transpose(mat(classLabels))
# matMerge = Untils.mergMatrix(mat(dataMat),classMat)

# 元素乘法
# a = mat([1,1,1]) ;b = mat([2,2,2])
# print multiply(a,b)

# 测试BackPropgation.dlogsig(hp,tau)
# A = mat([0,1,2]);
# print "A*(1-A)",multiply(A,(1-A))
