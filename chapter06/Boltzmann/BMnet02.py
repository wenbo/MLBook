# -*- coding:utf-8 -*-
# Filename : testBoltzmann01.py

import operator
import copy
import Untils
from BMNet import *
from numpy import *
import matplotlib.pyplot as plt 
bmNet = BoltzmannNet()
bmNet.loadDataSet("dataSet25.txt")
bmNet.train()
print "循环迭代",bmNet.iteration,"次"
print "最优解:",bmNet.bestdist
print "最佳路线:",bmNet.bestpath	

# 显示优化后城市图,路径图
bmNet.drawScatter(plt)
bmNet.drawPath(plt)
plt.show()
# 绘制误差算法收敛曲线
bmNet.TrendLine(plt)
plt.show()