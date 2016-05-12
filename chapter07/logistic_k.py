# -*- coding: utf-8  -*-

from numpy import *
import sys
import matplotlib.pyplot as plt 

def scatterplot(k,x1,x2):# 绘制图形
	fig = plt.figure();	
	ax1 = fig.add_subplot(111) 
	ax1.plot(x1)
	ax1.plot(x2)
	plt.title("k="+str(k)) 
	plt.show()	
def logistic_map(k,init):
	maxIter=100 # 最大迭代数
	x = range(maxIter)
	x[0]=init
	for i in xrange(maxIter-1):  
		x[i+1] = k*x[i]*(1.0-x[i])
	return x	
	
x1 = logistic_map(3.6,0.1)
x2 = logistic_map(3.6,0.9)
scatterplot(3.6,x1,x2)	
'''
x1 = logistic_map(3.5,0.1)
x2 = logistic_map(3.5,0.9)
scatterplot(3.5,x1,x2)	
'''