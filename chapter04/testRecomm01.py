# -*- coding: utf-8 -*-
# Filename : testRecomm01.py

from numpy import *
import numpy as np 
import operator
from svdRec import *
from Recommand_Lib import *
import matplotlib.pyplot as plt 

eps = 1.0e-6
# 加载修正后数据
dataMat = file2matrix("ml_data/training.txt","\t")
print dataMat[0][0]
# 相似公式：夹角余弦
output1 = recommend(dataMat,1)
print output1

# 相似公式：欧氏距离
# output2 = recommend(dataMat,1,simMeas=ecludSim)
# print output2

# 相似公式：相关系数
# output3 = recommend(dataMat,1,simMeas=pearsSim)
# print output3
