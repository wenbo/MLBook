# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

# 行数为row;列数为column 
row = 3
column = 3
#-----------------定义符号函数代码-----------------#
# 1. 初始化一个Theano张量
A = theano.shared(
	# numpy.ones((row, column)矩阵的值; dtype=浮点数类型：float32:	
	value=np.ones((row, column), dtype=theano.config.floatX),  
	name='A',  # 变量名
	borrow=True  # 与numpy共享array的内存空间
)

# 使用numpy array初始化
Xlist = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]
B = theano.shared( value=np.array(Xlist,dtype=theano.config.floatX), name = 'B' ,  borrow = True )
x = T.dmatrix('x')
y = T.dmatrix('y')
z = T.vector('z')
out = T.mean(x+y) #计算均值
myfunc1 = theano.function(  
	inputs=[],# 指定函数参数
	outputs=out, 
	givens=[(x,A),(y,B)]
)
# 遍历List
C = theano.shared(np.asarray([1.0,2.0,3.0,4.0,5.0],dtype=theano.config.floatX))
idx = T.lscalar('idx')
f1=T.sum(z) # 求和
myfunc2 = theano.function([idx],outputs=f1,givens={z:C[0:idx]})


#-----------------输出函数运行结果-----------------#
print A.get_value()
print np.shape(B.get_value())
print myfunc1()
print myfunc2(4)
