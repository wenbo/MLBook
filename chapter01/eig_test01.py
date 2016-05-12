# -*- coding: utf-8 -*-
# Filename : matrix05.py
import operator
from numpy import *

eps = 1.0e-6 # 误差量

# 矩阵的特征值和特征向量
A = mat([[8,1,6],[3,5,7],[4,9,2]])

# 手动计算特征值：
m,n = shape(A)
# Aeig = lambda*I-A = [[lambda-8,-1],[-6;-3,lambda-5,-7],[-4,-9,lambda-2]]
# (lambda-8)*(lambda-5)*(lambda-2)-190-24*(5-lambda)-3*(2-lambda)-63*(8-lambda)
equationA = [1,-15,-24,360] #得到系数方程矩阵
evals = roots(equationA) # 计算矩阵方程的根
print "特征值:" ,evals

evals, evecs = linalg.eig(A)
print "特征值:",evals,"\n特征向量:", evecs  

# 特征值和特征向量,还原原矩阵
sigma = evals*eye(m)
print evecs*sigma*linalg.inv(evecs)
