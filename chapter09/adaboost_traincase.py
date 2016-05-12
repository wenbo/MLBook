# -*- encoding: utf-8 -*-
# Filename : testAdaboost2.py

from numpy import *
import sys
from adaboostlib import *
# 导入训练集
dataArr,labelArr = loadDataSet('train.dat')

weakClassArr,aggClassEst = adaBoostTrain(dataArr,labelArr,numIt=10) # 训练分类器
print "weakClassArr:",weakClassArr # 输出弱分类器
# plotROC(aggClassEst.T, labelArr) # 绘制ROC曲线


