# -*- coding: utf-8 -*-
# Filename : testKohonen02.py

import numpy as np 
import operator
import Untils
import Kohonen
from numpy import *
import matplotlib.pyplot as plt 

# 加载坐标数据文件
dataSet = Untils.loadDataSet("dataset.txt");
dataMat = mat(dataSet)

# kohonen算法
classLabel = Kohonen.kohonen(dataMat)

# 去重
lst = unique(classLabel.tolist()[0])

# 绘图
i = 0;
for cindx in lst:
	myclass = nonzero(classLabel==cindx)[1]	
	xx = dataMat[myclass].copy()
	if i ==0:
		plt.plot(xx[:,0],xx[:,1],'bo')
	if i ==1:
		plt.plot(xx[:,0],xx[:,1],'r*')
	if i ==2:
		plt.plot(xx[:,0],xx[:,1],'gD')			
	if i ==3:
		plt.plot(xx[:,0],xx[:,1],'c^')						
	i +=1			
plt.show()

    	