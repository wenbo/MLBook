# -*- coding:utf-8 -*-
# Filename : testBoltzmann01.py

import operator
import copy
import Untils
import Boltzmann
from numpy import *
import matplotlib.pyplot as plt 

dataSet = Untils.loadDataSet("cities.txt")
cityPosition = mat(dataSet)
m,n = shape(cityPosition)
pn = m
# 将城市的坐标矩阵转换为邻接矩阵（城市间距离矩阵）
dist = Boltzmann.distM(cityPosition,cityPosition.transpose())

# 初始化
MAX_ITER = 2000 # 1000-2000
MAX_M = m;
Lambda = 0.97;
T0 = 1000; # 100-1000
# 构造一个初始可行解
x0 = arange(m)
random.shuffle(x0)
# 
T = T0;
iteration = 0;
x = x0;                   # 路径变量
xx = x0.tolist();         # 每个路径
di = []
di.append(Boltzmann.pathLen(dist, x0))   # 每个路径对应的距离
k = 0;                  # 路径计数

# 外循环
while iteration <= MAX_ITER:
	# 内循环迭代器
	m = 0;
	# 内循环
	while m <= MAX_M:
		# 产生新路径
		newx =  Boltzmann.changePath(x)
		# 计算距离
		oldl = Boltzmann.pathLen(dist,x)
		newl = Boltzmann.pathLen(dist,newx)
		if ( oldl > newl):   # 如果新路径优于原路径，选择新路径作为下一状态
			x = newx
			xx.append(x)    # xx[n,:] = x
			di.append(newl) # di[n] = newl			
			k += 1            
		else: # 如果新路径比原路径差，则执行概率操作
			tmp = random.rand()
			sigmod = exp(-(newl - oldl)/T)
			if tmp < sigmod:
				x = newx
				xx.append(x)    # xx[n,:] = x
				di.append(newl) # di[n]= newl				
				k += 1
		m += 1            # 内循环次数加1
	# 内循环
	iteration += 1      # 外循环次数加1
	T = T*Lambda        # 降温

# 计算最优值
bestd = min(di) 
indx = argmin(di)
bestx = xx[indx]
print "循环迭代",k,"次"
print "最优解:",bestd
print "最佳路线:",bestx	

# 优化前城市图,路径图
Untils.drawScatter(cityPosition,flag=False)
Untils.drawPath(range(m-1),cityPosition)

# 显示优化后城市图,路径图
Untils.drawScatter(cityPosition,flag=False)
Untils.drawPath(bestx,cityPosition,color='r')

# 绘制误差趋势线
x0 = range(len(di));
Untils.TrendLine(x0,di)
