# -*- coding: UTF-8 -*-
# Filename : BPTest.py

from numpy import *
import operator
from bpNet import *
import matplotlib.pyplot as plt 

# 数据集
bpnet = BPNet() 
bpnet.loadDataSet("testSet2.txt")
bpnet.dataMat = bpnet.normalize(bpnet.dataMat)

# 绘制数据集散点图
bpnet.drawClassScatter(plt)

# BP神经网络进行数据分类
bpnet.bpTrain()

print bpnet.out_wb
print bpnet.hi_wb

# 计算和绘制分类线
x,z = bpnet.BPClassfier(-3.0,3.0)
bpnet.classfyLine(plt,x,z)
plt.show()
# 绘制误差曲线
bpnet.TrendLine(plt)
plt.show()
