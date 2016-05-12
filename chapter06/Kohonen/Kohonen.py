# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt
class Kohonen(object):
	def __init__(self):	
		self.lratemax=0.8;  # 最大学习率	
		self.lratemin=0.05  # 最小学习率	
		self.rmax=5.0;      # 最大聚类半径
		self.rmin=0.5       # 最小聚类半径
		self.Steps = 1000    # 迭代次数
		self.lratelist=[]
		self.rlist=[]
		self.w=[]
		self.M=2;	          # 二维聚类网格参数
		self.N=2            # 二维聚类网格参数
		self.dataMat=[]
		self.classLabel=[]
		
	def loadDataSet(self,fileName): # 加载数据文件
		numFeat = len(open(fileName).readline().split('\t')) - 1 
		fr = open(fileName)
		for line in fr.readlines():
			lineArr =[]
			curLine = line.strip().split('\t')
			lineArr.append(float(curLine[0]));
			lineArr.append(float(curLine[1]));
			self.dataMat.append(lineArr)
		self.dataMat = mat(self.dataMat)
		
	# 数据标准化(归一化):		# 标准化
	def normalize(self,dataMat):
		[m,n]=shape(dataMat)
		for i in xrange(n-1):
			dataMat[:,i] = (dataMat[:,i]-mean(dataMat[:,i]))/(std(dataMat[:,i])+1.0e-10)
		return dataMat	
	
	# 计算矩阵各向量之间的距离--欧氏距离	
	def distEclud(self,matA,matB):
		ma,na = shape(matA);
		mb,nb = shape(matB);
		rtnmat= zeros((ma,nb))
		for i in xrange(ma):
			for j in xrange(nb):
	 			rtnmat[i,j] = linalg.norm(matA[i,:]-matB[:,j].T) 
		return 	rtnmat
	# 学习率和学习半径函数	
	def ratecalc(self,indx):
		lrate = self.lratemax-(float(indx)+1.0)/float(self.Steps)*(self.lratemax-self.lratemin) 
		r = self.rmax-(float(indx)+1.0)/float(self.Steps)*(self.rmax-self.rmin)
		return lrate,r
	# 初始化第二层网格	
	def init_grid(self):
		k=0;# 构建第二层网格模型
		grid = mat(zeros((self.M*self.N ,2)));
		for i in xrange(self.M):
			for j in xrange(self.N):
				grid[k,:]=[i,j]
				k +=1;
		return grid		
# 主算法	
	def train(self):
    #1 构建输入层网络
		dm,dn = shape(self.dataMat) 
		normDataset = self.normalize(self.dataMat) # 归一化数据x
		#2 构建分类网格
		grid = self.init_grid() # 初始化第二层分类网格 
		#3 构建两层之间的权重向量
		self.w = random.rand(dn,self.M*self.N); #随机初始化权值 w
		distM = self.distEclud   # 确定距离公式
		#4 迭代求解
		if self.Steps < 10*dm:	self.Steps = 10*dm   # 设定最小迭代次数
		for i in xrange(self.Steps): 	
			lrate,r = self.ratecalc(i) # 计算学习率和分类半径
			self.lratelist.append(lrate);self.rlist.append(r)
			# 随机生成样本索引，并抽取一个样本
			k = random.randint(0,dm) 
			mySample = normDataset[k,:] 	
	
			# 计算最优节点：返回最小距离的索引值
			minIndx= (distM(mySample,self.w)).argmin()
			d1 = ceil(minIndx/self.M)   # 计算最近距离在第二层矩阵中的位置
			d2 = mod(minIndx,self.M)    
			distMat = distM(mat([d1,d2]),grid.T)
			nodelindx = (distMat<r).nonzero()[1] # 根据学习距离获取邻域内左右节点
			# 更新权重列
			for j in xrange(shape(self.w)[1]):
				if sum(nodelindx==j):
		 			self.w[:,j] = self.w[:,j]+lrate*(mySample[0]-self.w[:,j])
		# 分配类别标签
		self.classLabel = range(dm)
		for i in xrange(dm):
			self.classLabel[i] = distM(normDataset[i,:],self.w).argmin()
		self.classLabel = mat(self.classLabel)		
	
	def showCluster(self,plt):
		lst = unique(self.classLabel.tolist()[0]) # 去重
		# 绘图
		i = 0;
		for cindx in lst:
			myclass = nonzero(self.classLabel==cindx)[1]	
			xx = self.dataMat[myclass].copy()
			if i ==0:
				plt.plot(xx[:,0],xx[:,1],'bo')
			elif i ==1:
				plt.plot(xx[:,0],xx[:,1],'rd')
			elif i ==2:
				plt.plot(xx[:,0],xx[:,1],'gD')			
			elif i ==3:
				plt.plot(xx[:,0],xx[:,1],'c^')						
			i +=1			
		plt.show()
# 绘制趋势线: 可调整颜色		
	def TrendLine(self,plt,mylist,color='r'):
		X = linspace(0,len(mylist),len(mylist))
		Y = mylist
		plt.plot(X,Y,color)
		plt.show()