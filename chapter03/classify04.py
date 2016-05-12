# -*- coding: utf-8 -*-
# Filename : 07store_grab.py

from numpy import *
from math import log
from ID3DTree import * 
import copy
import treePlotter2
import matplotlib.pyplot as plt

dtree = ID3DTree()

labels = ["age","revenue","student","credit"]
vector = ['0','1','0','0'] # ['0','1','0','0','no']
mytree = dtree.grabTree("data.tree")
print "真实输出 ","no","->","决策树输出",dtree.predict(mytree,labels,vector)
