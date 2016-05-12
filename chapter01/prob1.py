# -*- coding: utf-8 -*-

import sys  
import os
import time
from numpy import *

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

featuremat = mat([[88.5,96.8,104.1,111.3,117.7,124.0,130.0,135.4,140.2,145.3,151.9,159.5,165.9,169.8,171.6,172.3,172.7],
[12.54,14.65,16.64,18.98,21.26,24.06,27.33,30.46,33.74,37.69,42.49,48.08,53.37,57.08,59.35,60.68,61.40]])

# 计算均值
mv1 = mean(featuremat[0]) # 第一列的均值
mv2 = mean(featuremat[1]) # 第二列的均值 
# 计算两列标准差
dv1 = std(featuremat[0])
dv2 = std(featuremat[1])

corref = mean(multiply(featuremat[0]-mv1,featuremat[1]-mv2))/(dv1*dv2)
print corref

print corrcoef(featuremat)

covinv = linalg.inv(cov(featuremat))
print covinv
tp = featuremat.T[0]-featuremat.T[1]
distma = sqrt(dot(dot(tp,covinv),tp.T))
print distma 

