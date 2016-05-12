# -*- coding: utf-8 -*-

import sys  
import os
import time
from numpy import *

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

vectormat = mat([[1,2,3],[4,5,6]])
v12 = vectormat[0]-vectormat[1]
print sqrt(v12*v12.T)
#norm
varmat = std(vectormat.T,axis=0)
normvmat = (vectormat-mean(vectormat))/varmat.T
#norm
print normvmat
normv12 = normvmat[0]-normvmat[1]
print sqrt(normv12*normv12.T)
