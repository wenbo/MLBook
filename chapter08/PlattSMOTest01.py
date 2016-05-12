# -*- coding: utf-8 -*-
# Filename : 02PlattSMO.py

from numpy import *
import numpy as np
from svmplatt import *
import matplotlib.pyplot as plt 

svm = PlattSVM()
svm.C=70  # 惩罚因子C: 0.6, 
svm.tol=0.001  # 容错率:0.001
svm.maxIter=200
svm.kValue['Gaussian']= 3.0 # 核函数
svm.loadDataSet('nolinear.txt')
# 主 platt smo 函数
svm.train()
# 根据拉格朗日alphas乘子计算W向量
print svm.svIndx
print shape(svm.sptVects)[0]
print "b:",svm.b
# print "lambdas[lambdas > 0]:",svm.lambdas[svm.lambdas > 0]
svm.scatterplot(plt)
# 显示绘制的图形
plt.show()