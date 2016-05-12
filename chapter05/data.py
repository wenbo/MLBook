# -*- coding: utf-8 -*-
import os
import sys
import numpy as np 
import operator
from numpy import *
from common_libs import *

# 1.导入数据
Input = file2matrix("testSet2.txt","\t")
m,n=shape(Input)
print m,n
newdata = zeros((m,3))
newdata[:,:2] = Input[:,:2]
newdata[:,2:2] = Input[:,3:3]
print newdata[:100,:]
