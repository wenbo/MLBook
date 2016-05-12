# -*- coding: utf-8 -*-

import sys  
import os
import time
from numpy import *

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

vector1 = mat([1,2,3])
vector2 = mat([4,7,5])
'''
print sqrt((vector1-vector2)*((vector1-vector2).T))

print sum(abs(vector1-vector2))

print abs(vector1-vector2).max()

#计算夹角余弦
Lv1 = sqrt(vector1*vector1.T)
Lv2 = sqrt(vector2*vector2.T)
cosV12 = vector1*vector2.T/(Lv1*Lv2)
print cosV12
'''