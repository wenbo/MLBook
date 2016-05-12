# -*- coding: utf-8 -*-

import sys  
import os
import time
from numpy import * 

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

mymatrix = mat([[1,2,3],[4,5,6],[7,8,9]])
[m,n]=shape(mymatrix) # 矩阵的行列数
print "矩阵的行数和列数:",m,n

myscl1 = mymatrix[0] # 按行切片
print "按行切片:",myscl1

myscl2 = mymatrix.T[0] # 按列切片
print "按列切片:",myscl2

mycpmat = mymatrix.copy() # 矩阵的复制
print "复制矩阵:\n",mycpmat

#比较
print "矩阵元素的比较:\n",mymatrix < mymatrix.T

# 矩阵的特征值和特征向量
A = [[8,1,6],[3,5,7],[4,9,2]]
evals, evecs = linalg.eig(A)
print "特征值:",evals,"\n特征向量:", evecs