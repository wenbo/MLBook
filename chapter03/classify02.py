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
# dtree.loadDataSet("lenses.txt",['age','prescript','astigmatic','tearRate'])
dtree.train()
print dtree.tree
treePlotter2.createPlot(dtree.tree)


