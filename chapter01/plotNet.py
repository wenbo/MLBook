# -*- coding: utf-8 -*-

import sys  
import os
import numpy as np
from numpy import *
import matplotlib.pyplot as plt 

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# nodelist = ["city1","city2","city3","city4","city5","city6","city7","city8"]
dist = mat([[0.1,0.1],[0.9,0.5],[0.9,0.1],[0.45,0.9],[0.9,0.8],[0.7,0.9],[0.1,0.45],[0.45,0.1]])
m,n = shape(dist)
# 绘图
fig = plt.figure()
ax = fig.add_subplot(111) 
ax.scatter(dist.T[0],dist.T[1],c='blue',marker='o',s=100)
for point in dist.tolist():
	plt.annotate("("+str(point[0])+", "+str(point[1])+")",xy = (point[0],point[1]))	
xlist = []
ylist = []
for px,py in zip(dist.T.tolist()[0],dist.T.tolist()[1]):
	xlist.append([px])
	ylist.append([py])
# print xlist
# print ylist
ax.plot(xlist,ylist,'r') 
plt.show()
