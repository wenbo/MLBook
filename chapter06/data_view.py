# -*- coding: utf-8 -*-
import os
import sys
import numpy as np 
import copy
from numpy import *
from common_libs import *
import matplotlib.pyplot as plt 

# 输入数据
Input = file2matrix("test.txt","\t")
target = Input[:,0].copy()
Input[:,0] = 	Input[:,2].copy()
Input[:,2] = 	target.copy()	
[m,n] = shape(Input) 

# 按分类绘制散点图
drawScatterbyLabel(plt,Input)

plt.show()
