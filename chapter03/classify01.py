# -*- coding: utf-8 -*-
# Filename : 07store_grab.py

from numpy import *
import trees2
import numpy as np 
import operator
from math import log
import copy

dataSet,labels = trees2.createDataSet()
print dataSet,labels
treelabels = copy.deepcopy(labels)
myTree = trees2.createTree(dataSet,treelabels)
print myTree
testVec=[1,0]
classLabel = trees2.classify(myTree,labels,testVec)
print "classLabel:",classLabel


