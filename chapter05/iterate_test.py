# -*- coding: utf-8 -*-
import os
import sys
import numpy as np 
from numpy import *
from common_libs import *
import matplotlib.pyplot as plt 
# 求解原方程
A=mat([[8,-3,2],[4,11,-1],[6,3,12]])
b=mat([20,33,36])
result= linalg.solve(A,b.T)
print result

# 迭代求原方程组的解：x(k+1)=B0*x(k)+f
B0 = mat([[0.0,3.0/8.0,-2.0/8.0],[-4.0/11.0,0.0,1.0/11.0],[-6.0/12.0,-3.0/12.0,0.0]])
m,n = shape(B0)  
f = mat([[20.0/8.0],[33.0/11.0],[36.0/12.0]])

error = 1.0e-6 # 误差阈值
steps = 100 # 迭代次数
xk = zeros((n,1)) # 初始化 xk=x0
errorlist =[] # 记录逐次逼近的误差列表
for k in xrange(steps): # 主程序
	xk_1 = xk     # 上一次的xk
	xk = B0*xk+f  # 本次xk
	errorlist.append(linalg.norm(xk-xk_1)) # 计算并存储误差
	if errorlist[-1]<error: # 判断误差是否小于阈值
		print k+1 # 输出迭代次数
		break		
print xk # 输出计算结果
# 绘制误差收敛散点图
matpts = zeros((2,k+1))
matpts[0] = linspace(1,k+1,k+1)
matpts[1] = array(errorlist)
drawScatter(plt,matpts) 
plt.show()
