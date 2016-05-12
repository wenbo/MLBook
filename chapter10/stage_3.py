# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 23:22:17 2014

@author: wencc
"""
import theano
import theano.tensor as T
from theano import function
import numpy as np

# 定义向量
XWB = [[3,1],[-1,-2],[-1,5]]
X = np.array(XWB[0],dtype=theano.config.floatX)
W = np.array(XWB[1],dtype=theano.config.floatX)
B = np.array(XWB[2],dtype=theano.config.floatX)


logX = T.log(X) # 计算对数 
print logX.eval()
meanW= T.mean(W) # 计算均值
print meanW.eval()
neqW= T.neq(W, B) # 返回X!=B的逻辑结果
print neqW.eval() 

x = T.dscalar('x')
y = x**2+10
grady = T.grad(y,x) # 计算梯度
f = function([x], grady)
print f(100)
# 计算softmax函数
Y_pred = T.nnet.softmax(T.dot(X, W) + B)
print Y_pred.eval()