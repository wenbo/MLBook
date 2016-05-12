# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
from Recommand_Lib import kNN
dataMat=mat([[0.417,0.0,0.25,0.333],[0.3,0.4,0.0,0.3],[0.0,0.0,0.625,0.375],[0.278,0.222,0.222,0.278],[0.263,0.211,0.263,0.263]])
testSet = [0.334,0.333,0.0,0.333]
classLabel = np.array(['B','C','D','E','F'])
print kNN(testSet,dataMat,classLabel,3)