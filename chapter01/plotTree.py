# -*- coding: utf-8 -*-

import sys  
import os
import numpy as np
import matplotlib.pyplot as plt
import treePlotter as tp 

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# 绘制树

myTree = {'root': {0: 'leaf node', 1: {'level 2': {0: 'leaf node', 1: 'leaf node'}},2:{'level2': {0: 'leaf node', 1: 'leaf node'}}}}
tp.createPlot(myTree)

