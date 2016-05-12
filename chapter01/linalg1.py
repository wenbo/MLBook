# -*- coding: utf-8 -*-

import sys  
import os
import time
from numpy import * 

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# n阶方阵的行列式运算
A = mat([[1,2,4,5,7,],[9,12,11,8,2,],[6,4,3,2,1,],[9,1,3,4,5],[0,2,3,4,1]])
'''
print "det(A):",linalg.det(A);  # 方阵的行列式

invA = linalg.inv(A) # 矩阵的逆
print "inv(A):",invA

AT = A.T   #矩阵的对称
print A*AT

#矩阵的秩
print linalg.matrix_rank(A)

#可逆矩阵求解
b = [1,0,1,0,1] 
S = linalg.solve(A,transpose(b))
print S
'''
