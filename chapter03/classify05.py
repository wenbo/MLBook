# -*- coding: utf-8 -*-

from numpy import *
from math import log
from C45DTree import * 
import treePlotter2

dtree = C45DTree()
dtree.loadDataSet("dataset.dat",["age","revenue","student","credit"])
dtree.train()
print dtree.tree
treePlotter2.createPlot(dtree.tree)