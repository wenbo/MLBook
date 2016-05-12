# -*- coding: utf-8 -*-
# Filename : testKohonen.py

import numpy as np 
import operator
import Untils
import Kohonen
from numpy import *
import matplotlib.pyplot as plt 

# 加载坐标数据文件
dataSet = Untils.loadDataSet("dataset.txt");
dataMat = mat(dataSet)
dm,dn = shape(dataMat)
# 归一化数据
normDataset = Kohonen.mapMinMax(dataMat)
# 参数
# 学习率
rate1max=0.8  #0.8
rate1min=0.05;
# 学习半径
r1max=3;
r1min=0.8 #0.8

## 网络构建
Inum=2;
M=2;
N=2;
K=M*N;          #Kohonen总节点数  
 
# Kohonen层节点排序
k=0;
jdpx = mat(zeros((K,2)));
for i in range(M):
    for j in range(N):
        jdpx[k,:]=[i,j];
        k=k+1;

# 权值初始化
w1 = random.rand(Inum,K); #第一层权值

## 迭代求解
ITER = 200
for i in range(ITER):
	
	#自适应学习率和相应半径
	rate1 = rate1max-(i+1)/float(ITER)*(rate1max-rate1min)
	r = r1max-(i+1)/float(ITER)*(r1max-r1min)
	# 随机抽取一个样本
	k = random.randint(0,dm) #生成样本的索引,不包括最高值
	myndSet = normDataset[k,:] #xx
		
	# 计算最优节点：返回最小距离的索引值
	minIndx= (Kohonen.distM(myndSet,w1)).argmin()
	d1 = ceil(minIndx/M)
	d2 = mod(minIndx,N)
	distMat = Kohonen.distM(mat([d1,d2]),jdpx.transpose())
	nodelindx = (distMat<r).nonzero()[1]
	for j in range(K):
		if sum(nodelindx==j):
			w1[:,j] = w1[:,j]+rate1*(myndSet.tolist()[0]-w1[:,j])

# 学习阶段
classLabel = range(dm);
for i in range(dm):
    classLabel[i] = Kohonen.distM(normDataset[i,:],w1).argmin()
# 去重
lst = unique(classLabel)
print lst
classLabel = mat(classLabel)
# 绘图
i = 0;
for cindx in lst:
	myclass = nonzero(classLabel==cindx)[1]	
	xx = dataMat[myclass].copy()
	if i ==0:
		plt.plot(xx[:,0],xx[:,1],'bo')
	if i ==1:
		plt.plot(xx[:,0],xx[:,1],'r*')
	if i ==2:
		plt.plot(xx[:,0],xx[:,1],'gD')			
	if i ==3:
		plt.plot(xx[:,0],xx[:,1],'c^')						
	i +=1			
plt.show()

    	