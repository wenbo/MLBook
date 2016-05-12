# -*- coding: utf-8 -*-
# Filename : Boltzmann.py

import operator
from numpy import *
import copy
import matplotlib.pyplot as plt 

class BoltzmannNet(object):
	def __init__(self):	
		self.dataMat = []
		self.MAX_ITER = 2000
		self.T0 = 1000
		self.Lambda = 0.97
		self.iteration = 0
		self.dist=[]
		self.pathindx=[]
		self.bestdist=0
		self.bestpath=[]
	# 加载数据文件
	def loadDataSet(self,fileName): 
		numFeat = len(open(fileName).readline().split('\t')) - 1 
		dataMat = []; 
		fr = open(fileName)
		for line in fr.readlines():
			lineArr =[]
			curLine = line.strip().split('\t')
			lineArr.append(float(curLine[0]));
			lineArr.append(float(curLine[1]));
			dataMat.append(lineArr)
		self.dataMat = mat(dataMat)			
	
	def distEclud(self,matA,matB): # 计算矩阵各向量之间的距离--欧氏距离	
		ma,na = shape(matA);	mb,nb = shape(matB);
		rtnmat= zeros((ma,nb))
		for i in xrange(ma):
			for j in xrange(nb):
	 			rtnmat[i,j] = linalg.norm(matA[i,:]-matB[:,j].T) 
		return 	rtnmat
	# 计算路径长度	
	def pathLen(self,dist,path):		
		N = len(path)  
		plen = 0;
		for i in xrange(0,N-1):  # 长度为N的向量，包含从1-N的整数
			plen += dist[path[i], path[i+1]]
		plen +=  dist[path[0], path[N-1]]
		return plen
	# 路径交换函数
	def changePath(self,old_path):	
		N = len(old_path)
		if random.rand() < 0.25:   # 产生两个位置，并交换
			chpos = floor(random.rand(1,2)*N).tolist()[0] 
			new_path = copy.deepcopy(old_path)
			new_path[int(chpos[0])] = old_path[int(chpos[1])]
			new_path[int(chpos[1])] = old_path[int(chpos[0])]  
		else: # 产生三个位置，交换a-b和b-c段
			d = ceil(random.rand(1,3)*N).tolist()[0]; d.sort() #随机生成路径
			a = int(d[0]); b = int(d[1]); c = int(d[2])	
			if a != b and b != c:
				new_path = copy.deepcopy(old_path)
				new_path[a:c-1] = old_path[b-1:c-1] + old_path[a:b-1]	
			else:
				new_path = self.changePath(old_path)
		return new_path
	# 玻尔兹曼函数
	def boltzmann(self,newl,oldl,T):
		return exp(-(newl - oldl)/T)
	# 初始化网络	
	def initBMNet(self,m,n,distMat):# 构造一个初始可行解
		self.pathindx = range(m)
		random.shuffle(self.pathindx); # 随机生成每个路径
		self.dist.append(self.pathLen(distMat, self.pathindx))   # 每个路径对应的距离		
		return self.T0,self.pathindx,m
	# 训练样本
	def train(self):
		[m,n] = shape(self.dataMat)		
		distMat = self.distEclud(self.dataMat,self.dataMat.T)  # 转换为邻接矩阵（距离矩阵）
		# T:当前温度，curpath当前路径索引	MAX_M内循环最大迭代次数		
		[T,curpath,MAX_M]=self.initBMNet(m,n,distMat)  
		step=0; # 初始化外循环迭代		
		while step <= self.MAX_ITER:		# 外循环	
			m = 0; # 内循环计数器			
			while m <= MAX_M:  # 内循环
				curdist = self.pathLen(distMat,curpath)    # 计算当前路径距离
				newpath = self.changePath(curpath)         # 产生新路径	
				newdist = self.pathLen(distMat,newpath)    # 计算新路径距离
				if ( curdist > newdist):   # 如果新路径优于原路径，选择新路径作为下一状态
					curpath = newpath
					self.pathindx.append(curpath)
					self.dist.append(newdist)			
					self.iteration += 1            
				else: # 如果新路径比原路径差，则执行随机操作 
					if random.rand() < self.boltzmann(newdist,curdist,T) :
						curpath = newpath
						self.pathindx.append(curpath)
						self.dist.append(newdist)		
						self.iteration += 1
				m += 1
			step += 1
			T = T*self.Lambda        # 降温
		# 计算最优值
		self.bestdist = min(self.dist) 
		indxes = argmin(self.dist)
		self.bestpath = self.pathindx[indxes]
	#绘制路径	
	def drawPath(self,plt,color='b'):
		m,n = shape(self.dataMat)	
		px = (self.dataMat[self.bestpath,0]).tolist()
		py = (self.dataMat[self.bestpath,1]).tolist()
		px.append(px[0]); py.append(py[0])
		plt.plot(px,py,color) 
	# 绘制散点 
	def drawScatter(self,plt):
		px = (self.dataMat[:,0]).tolist()
		py = (self.dataMat[:,1]).tolist()	
		plt.scatter(px,py,c='green',marker='o',s=60)
		i=65
		for x,y in zip(px,py):
			plt.annotate(str(chr(i)),xy =(x[0]+40,y[0]),color='black')	
			i += 1		
	# 绘制趋势线
	def TrendLine(self,plt,color='b'):
		plt.plot(range(len(self.dist)),self.dist,color)
