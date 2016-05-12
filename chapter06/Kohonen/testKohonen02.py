# -*- coding: utf-8 -*-
# Filename : testKohonen02.py

import numpy as np 
from Kohonen import *
from numpy import *
import matplotlib.pyplot as plt 
# 矩阵各元素平方之和
def errorfunc(inX):
	return sum(power(inX,2))*0.5
# 加载坐标数据文件
SOMNet = Kohonen()
SOMNet.loadDataSet("dataset2.txt");
SOMNet.train()
print SOMNet.w
SOMNet.showCluster(plt)
SOMNet.TrendLine(plt,SOMNet.lratelist)
SOMNet.TrendLine(plt,SOMNet.rlist)


