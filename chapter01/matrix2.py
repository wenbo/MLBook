# -*- coding: utf-8 -*-

import sys  
import os
import time
from numpy import *

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

'''
myOnes = ones([3,3]) #3*3的全1矩阵 
myEye = eye(3) # 3*3的单位阵
print myOnes+myEye
print myOnes-myEye
'''
mylist = [[1,2,3],[4,5,6],[7,8,9]]
mymatrix = mat(mylist)

a = 10
print a*mymatrix

print sum(mymatrix)

mymatrix2 = 1.5*ones([3,3])
print multiply(mymatrix,mymatrix2)

print power(mymatrix,2)