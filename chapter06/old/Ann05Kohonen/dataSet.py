# -*- coding: utf-8 -*-
# Filename : dataSet.py

import numpy as np 
import operator
import Untils
import Kohonen
from numpy import *
import matplotlib.pyplot as plt 

# 加载坐标数据文件
dataSet = Untils.loadDataSet("dataset.txt");
dataMat = mat(dataSet)
# print dataMat
normDataset = Kohonen.mapMinMax(dataMat)
# print normDataset

# 生成int随机数，不包含高值
# print random.randint(0,30)

# 计算向量中最小值的索引值
xx = mat([1,9])
w1 = mat([[1,2,3,4],[5,6,7,8]])
minIndx = Kohonen.distM(xx,w1).argmin()

# 计算距离
jdpx = mat([[0,0],[0,1],[1,0],[1,1]])
d1 = ceil(minIndx/4)
d2 = mod(minIndx,4)
mydist = Kohonen.distM(mat([d1,d2]),jdpx.transpose())
# print mydist



