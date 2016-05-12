# -*- coding: utf-8 -*-
# Filename : testRecomm01.py

from numpy import *
import numpy as np 
import operator
from svdRec import *
import matplotlib.pyplot as plt 

eps = 1.0e-6
# 夹角余弦，避免除0
def cosSim(inA,inB):
    denom = linalg.norm(inA)*linalg.norm(inB)
    return float(inA*inB.T)/(denom+eps)
    
# 加载修正后数据
A = mat([[5, 5, 3, 0, 5, 5],[5, 0, 4, 0, 4, 4],[0, 3, 0, 5, 4, 5],[5, 4, 3, 3, 5, 5]])
new = mat([[5,5,0,0,0,5]])
U,S,VT = linalg.svd(A.T)
V =VT.T
Sigma = diag(S)
r = 2  # 取前两个奇异值
# 近似后的U,S,V值
Ur = U[:,:r]
Sr = Sigma[:r,:r]
Vr = V[:,:r]
# 计算new的坐标值	
newresult = new*Ur*linalg.inv(Sr)
print newresult

maxv = 0 # 最大的余弦值
maxi = 0 # 最大值的下标
indx= 0 
# 计算最近似的结果
for vi in Vr:
	temp = cosSim(newresult,vi)
	if temp > maxv:  
	 	maxv = temp
	 	maxi = indx
	indx +=1
print maxv,maxi 	

