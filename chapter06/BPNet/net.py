# -*- coding: UTF-8 -*-
# Filename : 03BPTest.py

from numpy import *
import operator
import BackPropgation
import matplotlib.pyplot as plt 

# 数据集
dataSet,classLabels = BackPropgation.loadDataSet("testSet2.txt") # 初始化时第1列为全1向量, studentTest.txt
dataSet = mat(dataSet)
m,n=shape(dataSet) 
SampIn = dataSet.T
hi_wb = ones((m,1)) 
hi_input = SampIn*hi_wb
print hi_input