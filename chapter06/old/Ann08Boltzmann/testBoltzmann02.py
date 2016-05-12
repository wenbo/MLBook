# -*- coding:utf-8 -*-
# Filename : testBoltzmann01.py

import operator
import copy
import Untils
import Boltzmann
from numpy import *
import matplotlib.pyplot as plt 

dataSet = Untils.loadDataSet("dataSet25.txt")
cityPosition = mat(dataSet)
m,n = shape(cityPosition)
bestx,di = Boltzmann.boltzmann(cityPosition,MAX_ITER = 2000,T0 = 100)


# 优化前城市图,路径图
Untils.drawScatter(cityPosition,flag=False)
Untils.drawPath(range(m),cityPosition)

# 显示优化后城市图,路径图
Untils.drawScatter(cityPosition,flag=False)
Untils.drawPath(bestx,cityPosition,color='r')

# 绘制误差趋势线
x0 = range(len(di));
Untils.TrendLine(x0,di)
