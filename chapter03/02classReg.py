# -*- coding: utf-8 -*-
# Filename : 02classReg.py

from numpy import *
import numpy as np 
import operator
import regTrees2
import treeExplore
# 主程序
# 加载数据集
fileName = "ex0.txt"
dataSet = regTrees2.loadDataSet(fileName)
# 转换为矩阵
dataSet = mat(dataSet)

# 创建树
regTree = regTrees2.createTree(dataSet)
print "regTree:",regTree

treeExplore.createPlot(regTree)  # 创建决策树图