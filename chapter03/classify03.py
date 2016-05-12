# -*- coding: utf-8 -*-
# Filename : 07store_grab.py

from numpy import *
from math import log
from ID3DTree import * 
import copy
import treePlotter2
import matplotlib.pyplot as plt

dtree = ID3DTree()
dtree.loadDataSet("dataset.dat",["age","revenue","student","credit"])
dtree.train()

dtree.storeTree(dtree.tree,"data.tree")
mytree = dtree.grabTree("data.tree")
print mytree
