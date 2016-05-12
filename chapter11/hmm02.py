# -*- coding: utf-8 -*-
# Filename : hmm02.py

from numpy import *
# 起始概率
startP = mat([0.63,0.17,0.20])
# 状态转移概率[i,j]:i(t),j(t+1)
stateP = mat([[0.5,0.375,0.125],[0.25,0.125,0.675],[0.25,0.375,0.375]]) 
# 发射（混合）概率
emitP = mat([[0.6,0.20,0.15,0.05],[0.25,0.25,0.25,0.25],[0.05,0.10,0.35,0.50]])

# 计算概率：干旱－干燥－潮湿
state1Emit = multiply(startP.T,emitP[:,0])
print state1Emit
best = state1Emit.argmax()
print "max",state1Emit.max(),"path1:",state1Emit.argmax()

# 计算干燥的概率:
print state1Emit[best],stateP
state2Mat = multiply(state1Emit[best],stateP)
print state2Mat
state2Mat = dot(state2Mat,emitP[:,1])
print "max",state2Mat.max(),"path1:",state2Mat.argmax()
'''
# 计算潮湿的概率:
state3Mat = multiply(state2Mat[best],stateP)
state3Mat = dot(state3Mat,emitP[:,1])
print "max",state3Mat.max(),"path1:",state3Mat.argmax()
'''
