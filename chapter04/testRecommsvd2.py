# -*- coding: utf-8 -*-
# Filename : testRecomm01.py

from numpy import *
import numpy as np 
import operator
from Recommand_Lib import *
    
# 加载修正后数据
A = mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],[0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
         [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],[3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
         [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],[0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
         [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],[0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
         [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],[0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],])
new = mat([[1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])
indx,resultarray= recommand(A,new,r=2,rank=2,distCalc=cosSim)
print indx
print resultarray

