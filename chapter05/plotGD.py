# -*- coding: utf-8 -*-
'''
Created on Oct 28, 2010

@author: Peter
'''
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025 # 步长
x = np.arange(-2.0, 2.0, delta) #x轴取值
y = np.arange(-2.0, 2.0, delta) #y轴取值
X, Y = np.meshgrid(x, y) #绘制网格图
Z1 = -((X-1)**2)
Z2 = -(Y**2)
#Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
#Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 1.0 * (Z2 + Z1)+5.0 # 计算获取Z轴取值


plt.figure()
CS = plt.contour(X, Y, Z)
# 绘制点到点之间的箭头
plt.annotate('', xy=(0.05, 0.05),  xycoords='axes fraction',
             xytext=(0.2,0.2), textcoords='axes fraction',
             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args )
plt.text(-1.9, -1.8, 'P0')  # P0点位置
plt.annotate('', xy=(0.2,0.2),  xycoords='axes fraction',
             xytext=(0.35,0.3), textcoords='axes fraction',
             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args )
plt.text(-1.35, -1.23, 'P1') # P1点位置
plt.annotate('', xy=(0.35,0.3),  xycoords='axes fraction',
             xytext=(0.45,0.35), textcoords='axes fraction',
             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args )
plt.text(-0.7, -0.8, 'P2') # P2点位置
plt.text(-0.3, -0.6, 'P3') # P3点位置
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Gradient Ascent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
