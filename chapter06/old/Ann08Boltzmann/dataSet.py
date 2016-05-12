# -*- coding: utf-8 -*-
# Filename : dataSet.py

import operator
import Untils
import Boltzmann
from numpy import *
import matplotlib.pyplot as plt 

# 载入数据
dataSet = Untils.loadDataSet("dataset.txt")
cityPosition = mat(dataSet)
m,n = shape(cityPosition)

# 随机数
tmp = random.rand()
print tmp

# 不重复的随机数
path = mat(zeros((m,m)))
tmp = arange(m)
for i in range(m):	
	random.shuffle(tmp)
	path[i,:] = tmp
# print path

# 绘制数据散点图
# Untils.drawScatter(dataSet)

# 将城市的坐标矩阵转换为邻接矩阵（城市间距离矩阵）
dist = Boltzmann.distM(cityPosition,cityPosition.transpose())
path = arange(m)
# print Boltzmann.pathLen(dist,path)

# sort
a = [1,3,5,7,9,2,4,6,8,0]
a.sort()
# print a
x1 = 3; x2 = 5
# print a[x1:x2]
# print x2-x1	

a = [1,3,5,7,9,2,4,6,8,0]
b = a
c = a
a[2:8] = b[3:9]+[]
print a	
print b	
print c
# 改变路径
newPath = Boltzmann.changePath(path.tolist())
# print newPath

b = []
b.append(3.14)
b.append(5.78)
# print b



