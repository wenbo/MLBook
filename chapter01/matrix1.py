# -*- coding: utf-8 -*-

import sys  
import os
import time
import numpy as np 

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')


myZero = np.zeros([3,5]) #3*5的全零矩阵 
print myZero


myOnes = np.ones([3,5]) #3*5的全零矩阵 
print myOnes

# 随机矩阵:3行4列的0~1之间的随机数矩阵
myRand = np.random.rand(3,4)
print myRand

# 单位阵
myEye = np.eye(3) # 3*3的单位阵
print myEye
