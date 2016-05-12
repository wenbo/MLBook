# -*- coding: utf-8 -*-

from numpy import *
from math import log
from C45DTree import * 
import treePlotter2

dtree = C45DTree()
labels = ["age","revenue","student","credit"]
dtree.loadDataSet("dataset.dat",labels)
dtree.decisionTree()
vector = ['0','1','0','0'] # ['0','1','0','0','no']
print "真实输出 ","no","->","决策树输出",dtree.predict(dtree.tree,labels,vector)