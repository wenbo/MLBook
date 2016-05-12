# -*- coding: utf-8 -*-
# Filename : Kohonen.py

import operator
import Untils
from numpy import *
import matplotlib.pyplot as plt

# 归一化数据集
def mapMinMax(dataMat):
	ymin = -1; ymax = 1
	m,n = shape(dataMat)
	rtnMat = mat(zeros((m,n)))
	for i in range(n):
		xmin = dataMat[:,i].min()
		xmax = dataMat[:,i].max()
		rtnMat[:,i] = (ymax-ymin)*(dataMat[:,i]-xmin)/(xmax-xmin) + ymin;		
	return rtnMat;
	
# 计算矩阵各向量之间的距离:返回一个对称的n*n矩阵	
def distM(matA,matB):
	ma,na = shape(matA);
	mb,nb = shape(matB);
	rtnmat= zeros((ma,nb))
	for i in range(ma):
		for j in range(nb):
			rtnmat[i,j] = sqrt(sum(power(matA[i,:] - matB[:,j].transpose(),2)))
	return 	rtnmat

# 主算法	
def kohonen(dataMat,M=2,N=2,ITER = 200):
	dm,dn = shape(dataMat)
	# 归一化数据
	normDataset = mapMinMax(dataMat)
	# 参数
	# 学习率
	rate1max=0.8; rate1min=0.05
	# 学习半径
	r1max=3; r1min=0.8
  
	## 网络构建
	Inum = 2;	K=M*N          #Kohonen总节点数  
   
	# Kohonen层节点排序
	k=0;
	jdpx = mat(zeros((K,2)));
	for i in range(M):
		for j in range(N):
			jdpx[k,:]=[i,j]
			k=k+1;
	
	# 权值初始化
	w1 = random.rand(Inum,K); #第一层权值
  
	## 迭代求解
	for i in range(ITER):  	
		#自适应学习率和相应半径
		rate1 = rate1max-(i+1)/float(ITER)*(rate1max-rate1min)
		r = r1max-(i+1)/float(ITER)*(r1max-r1min)
		# 随机抽取一个样本
		k = random.randint(0,dm) # 生成样本的索引,不包括最高值
		myndSet = normDataset[k,:] #xx
			
		# 计算最优节点：返回最小距离的索引值
		minIndx= (distM(myndSet,w1)).argmin()
		d1 = ceil(minIndx/M)
		d2 = mod(minIndx,M)
		distMat = distM(mat([d1,d2]),jdpx.transpose())
		nodelindx = (distMat<r).nonzero()[1]
		for j in range(K):
			if sum(nodelindx==j):
				w1[:,j] = w1[:,j]+rate1*(myndSet.tolist()[0]-w1[:,j])
  
	# 学习阶段
	classLabel = range(dm)
	for i in range(dm):
		classLabel[i] = distM(normDataset[i,:],w1).argmin()

  # 返回类别标签
	return mat(classLabel)		