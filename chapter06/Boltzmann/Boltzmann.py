# -*- coding: utf-8 -*-
# Filename : Boltzmann.py

import operator
import Untils
from numpy import *
import copy
import matplotlib.pyplot as plt 

# 计算矩阵各向量之间的距离:返回一个对称的n*n矩阵	
def distM(matA,matB):
	ma,na = shape(matA);
	mb,nb = shape(matB);
	rtnmat= zeros((ma,nb))
	for i in xrange(ma):
		for j in xrange(nb):
			rtnmat[i,j] = sqrt(sum(power(matA[i,:] - matB[:,j].T,2)))
	return 	rtnmat
	
def pathLen(dist,path):
	# dist:N*N邻接矩阵
	# 长度为N的向量，包含从1-N的整数
	N = len(path)
	plen = 0;
	for i in xrange(0,N-1):
		plen += dist[path[i], path[i+1]]
	plen +=  dist[path[0], path[N-1]]
	return plen
	
def changePath(old_path):
	# 在oldpath附近产生新的路径
	if type(old_path) is not list :
		old_path = old_path.tolist()
	N = len(old_path)
	if random.rand() < 0.25:   # 产生两个位置，并交换
		chpos = floor(random.rand(1,2)*N) # random.rand(1,2)
		chpos = chpos.tolist()[0]
		new_path = copy.deepcopy(old_path)
		new_path[int(chpos[0])] = old_path[int(chpos[1])]
		new_path[int(chpos[1])] = old_path[int(chpos[0])]  
	else: # 产生三个位置，交换a-b和b-c段
		d = ceil(random.rand(1,3)*N);
		d = d.tolist()[0]
		d.sort()
		a = int(d[0]); b = int(d[1]); c = int(d[2])	
		if a != b and b != c:
			new_path = copy.deepcopy(old_path)
			new_path[a:c-1] = old_path[b-1:c-1] + old_path[a:b-1]	
		else:
			new_path = changePath(old_path)
	return new_path
	
def boltzmann(cityPosition,MAX_ITER = 2000,T0 = 1000,Lambda = 0.97):
  m,n = shape(cityPosition)
  pn = m
  # 将城市的坐标矩阵转换为邻接矩阵（城市间距离矩阵）
  dist = distM(cityPosition,cityPosition.T)  
  # 初始化
  MAX_M = m;
  # 构造一个初始可行解
  x0 = arange(m)
  random.shuffle(x0)
  # 
  T = T0;
  iteration = 0;
  x = x0;                   # 路径变量
  xx = x0.tolist();         # 每个路径
  di = []
  di.append(pathLen(dist, x0))   # 每个路径对应的距离
  k = 0;                  # 路径计数
  
  # 外循环
  while iteration <= MAX_ITER:
  	# 内循环迭代器
  	m = 0;
  	# 内循环
  	while m <= MAX_M:
  		# 产生新路径
  		newx =  changePath(x)
  		# 计算距离
  		oldl = pathLen(dist,x)
  		newl = pathLen(dist,newx)
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
  return bestx,di	