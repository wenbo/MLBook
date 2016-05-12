# -*- coding: utf-8 -*-

from numpy import *
import operator
from time import sleep

class PlattSVM(object):
	def __init__(self):
		self.X = []   # 输入数据集
		self.labelMat = []
		self.C = 0.0    # 惩罚因子
		self.tol = 0.0  # 容错率toler
		self.b = 0.0    # 截距初始值
		self.kValue={}
		self.maxIter=10000
		self.svIndx=[] # 支持向量下标
		self.sptVects=[]  # 支持向量
		self.SVlabel=[] # 支持类别标签

	# 加载数据集并初始化相应参数
	def loadDataSet(self,fileName):
		fr = open(fileName)
		for line in fr.readlines():
			lineArr = line.strip().split('\t')
			self.X.append([float(lineArr[0]), float(lineArr[1])])
			self.labelMat.append(float(lineArr[2]))
		self.initparam()	
		# 核函数列表
	def kernels(self,dataMat,A):
		m,n = shape(dataMat) 
		K = mat(zeros((m,1)))
		if self.kValue.keys()[0]=='linear': 
			K = dataMat * A.T   #linear kernel
		elif self.kValue.keys()[0]=='Gaussian':   # 高斯核
			for j in xrange(m):
				deltaRow = dataMat[j,:] - A
				K[j] = deltaRow*deltaRow.T
			K = exp(K/(-1*self.kValue['Gaussian']**2)) 
		else: raise NameError('无法识别的核函数')
		return K
	def initparam(self):
		self.X = mat(self.X)                  # 数据集
		self.labelMat = mat(self.labelMat).T	# 类别标签
		self.m = shape(self.X)[0]             # 数据集行数
		self.lambdas = mat(zeros((self.m,1))) # 拉格朗日乘子		
		self.eCache = mat(zeros((self.m,2)))  # 误差缓存
		self.K = mat(zeros((self.m,self.m)))  # 存储用于核函数计算的向量 
		for i in xrange(self.m):
			self.K[:,i] = self.kernels(self.X,self.X[i,:]) # kValue
	# 随机选择一个不等于i的j
	def randJ(self,i):
		j=i 
		while(j==i):
			j = int(random.uniform(0,self.m))
		return j
	
	# 调整大于H,小于L的aj
	def clipLambda(self,aj,H,L):
		if aj > H: aj = H
		if L > aj: aj = L
		return aj
		
	def calcEk(self,k):
		return float(multiply(self.lambdas,self.labelMat).T*self.K[:,k] + self.b) - float(self.labelMat[k])
	
	#选择lambda,从缓存中寻找具有最大误差的行索引作为j
	def chooseJ(self,i,Ei):
		maxK = -1; maxDeltaE = 0; Ej = 0
		self.eCache[i] = [1,Ei]  #更新误差缓存		
		validEcacheList = nonzero(self.eCache[:,0].A)[0] 
		if (len(validEcacheList)) > 1:
			for k in validEcacheList:
				if k == i: continue
				Ek = self.calcEk(k)
				deltaE = abs(Ei - Ek)
				if (deltaE > maxDeltaE):
					maxK = k; maxDeltaE = deltaE; Ej = Ek
			return maxK, Ej
		else:
				j = self.randJ(i)
				Ej = self.calcEk(j)
		return j, Ej
		
	# 主函数-内循环       
	def innerLoop(self,i):
		Ei = self.calcEk(i) # 计算和更新i的误差缓存
		# 如果误差超出容错率和错误分类允许的边界
		if ((self.labelMat[i]*Ei < -self.tol) and (self.lambdas[i] < self.C)) or ((self.labelMat[i]*Ei > self.tol) and (self.lambdas[i] > 0)):
			j,Ej = self.chooseJ(i, Ei) #选择具有最大误差的j
			lambdaIold = self.lambdas[i].copy(); lambdaJold = self.lambdas[j].copy();			
			if (self.labelMat[i] != self.labelMat[j]): # 见第二章公式十一
				L = max(0, self.lambdas[j] - self.lambdas[i])
				H = min(self.C, self.C + self.lambdas[j] - self.lambdas[i])
			else:
				L = max(0, self.lambdas[j] + self.lambdas[i] - self.C)
				H = min(self.C, self.lambdas[j] + self.lambdas[i])
			if L==H: 	return 0
			eta = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j,j] # 松弛变量，见第二章公式十五中目标函数的二阶导数
			if eta >= 0: 	return 0
			self.lambdas[j] -= self.labelMat[j]*(Ei - Ej)/eta # 见第二章公式九
			self.lambdas[j] = self.clipLambda(self.lambdas[j],H,L) # 见第二章公式十和公式十二
			self.eCache[j] = [1,self.calcEk(j)]	#计算和更新j的缓存
			if (abs(self.lambdas[j] - lambdaJold) < 0.00001): 	return 0
			self.lambdas[i] += self.labelMat[j]*self.labelMat[i]*(lambdaJold - self.lambdas[j]) 
			self.eCache[i] = [1,self.calcEk(i)]	#计算和更新j的缓存
			#见第二章公式十四
			b1 = self.b - Ei- self.labelMat[i]*(self.lambdas[i]-lambdaIold)*self.K[i,i] - self.labelMat[j]*(self.lambdas[j]-lambdaJold)*self.K[i,j]
			b2 = self.b - Ej- self.labelMat[i]*(self.lambdas[i]-lambdaIold)*self.K[i,j] - self.labelMat[j]*(self.lambdas[j]-lambdaJold)*self.K[j,j]
			# 根据KKT条件更新b的取值
			if (0 < self.lambdas[i]) and (self.C > self.lambdas[i]): self.b = b1
			elif (0 < self.lambdas[j]) and (self.C > self.lambdas[j]): self.b = b2
			else: self.b = (b1 + b2)/2.0
			return 1
		else: return 0
		
	# 主函数-外循环
	def train(self):    #full Platt SMO
		step = 0		
		entireflag = True; lambdaPairsChanged = 0 # entireflag全集扫描标志位
		# 外循环迭代器
		# 终止条件:超过最大迭代次数时,或未对lambda做出调整时退出
		while (step < self.maxIter) and ((lambdaPairsChanged > 0) or (entireflag)):
			lambdaPairsChanged = 0
			if entireflag: # 遍历整个数据集
				for i in xrange(self.m):
					lambdaPairsChanged += self.innerLoop(i) # 进入内循环
				step += 1
			else: # 遍历非边界的lambdas
				nonBoundIs = nonzero((self.lambdas.A > 0) * (self.lambdas.A < self.C))[0] # 通过KKT确定lambdas的位置
				for i in nonBoundIs:
					lambdaPairsChanged += self.innerLoop(i) # 进入内循环
				step += 1
			if entireflag: entireflag = False # 转换标志位 切换到两种遍历方式的另一种
			elif (lambdaPairsChanged == 0): entireflag = True  # 转换标志位 遍历整个数据集
		self.svIndx = nonzero(self.lambdas.A>0)[0]		# 输出计算后的支持向量索引
		self.sptVects = self.X[self.svIndx]            # 计算完成的支持向量
		self.SVlabel = self.labelMat[self.svIndx]      # 计算完成后的支持向量的类别标签
	# 计算权重向量
	def calcWs(self):
		m,n = shape(self.X)
		w = zeros((n,1))
		for i in xrange(m):
				w += multiply(self.lambdas[i]*self.labelMat[i],self.X[i,:].T)
		return w
	# 绘制散点图
	def scatterplot(self,plt):
		fig = plt.figure()
		ax = fig.add_subplot(111) 
		for i in xrange(shape(self.X)[0]):
			if self.lambdas[i] != 0: # KKT条件
				ax.scatter(self.X[i,0],self.X[i,1],c='green',marker='s',s=50)		
			elif self.labelMat[i] == 1:
				ax.scatter(self.X[i,0],self.X[i,1],c='blue',marker='o')
			elif self.labelMat[i] == -1:
				ax.scatter(self.X[i,0],self.X[i,1],c='red',marker='o')
	# 分类器
	def classify(self,testSet,testLabel):		
		errorCount = 0
		testMat = mat(testSet)
		m,n = shape(testMat)
		for i in xrange(m): # 用核函数划分测试集
			kernelEval = self.kernels(self.sptVects,testMat[i,:])
			predict= kernelEval.T * multiply(self.SVlabel,self.lambdas[self.svIndx]) + self.b
			if sign(predict)!=sign(testLabel[i]):  errorCount += 1
		return float(errorCount)/float(m)
