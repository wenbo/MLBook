# -*- coding: utf-8 -*-

import sys  
import os
import time
import numpy as np 

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

mylist = [1,2,3,4,5]
a = 10
mymatrix = np.mat(mylist)
print a*mymatrix

