# -*- coding: utf-8 -*-
# Filename : testRecomm01.py

from numpy import *
import numpy as np 
import operator
from svdRec import *
import matplotlib.pyplot as plt 

eps = 1.0e-6
# 加载修正后数据
A = mat([[5, 5, 3, 0, 5, 5],[5, 0, 4, 0, 4, 4],[0, 3, 0, 5, 4, 5],[5, 4, 3, 3, 5, 5]])

# 手工分解求矩阵的svd
U = A*A.T
lamda,hU = linalg.eig(U) # hU:U特征向量
VT = A.T*A
eV,hVT = linalg.eig(VT)  # hVT:VT特征向量
hV = hVT.T
# print "hU:",hU
# print "hV:",hV		
sigma = 	sqrt(lamda)         # 特征值
print "sigma:",sigma




Sigma = np.zeros([shape(A)[0], shape(A)[1]])
U,S,VT = linalg.svd(A)
# Sigma[:shape(A)[0], :shape(A)[0]] = np.diag(S)
# print U
print S
# print VT

# print U*Sigma*VT

