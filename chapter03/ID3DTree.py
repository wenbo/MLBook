# -*- coding: utf-8 -*-

from numpy import *
import math
import copy
import cPickle as pickle

class ID3DTree(object):
	def __init__(self):
		self.tree={}
		self.dataSet=[]
		self.labels=[]
	
	def loadDataSet(self,path,labels):
		recordlist = []
		fp = open(path,"rb") 	# 读取文件内容
		content = fp.read()
		fp.close()
		rowlist = content.splitlines() 	# 按行转换为一维表
		recordlist=[row.split("\t") for row in rowlist if row.strip()]	
		self.dataSet = recordlist
		self.labels = labels
		
	def train(self):
		labels = copy.deepcopy(self.labels)
		self.tree = self.buildTree(self.dataSet,labels)
			
	# 创建决策树主程序
	def buildTree(self,dataSet,labels):	  
		cateList = [data[-1] for data in dataSet] # 抽取源数据集的决策标签列
		# 程序终止条件1	: 如果classList只有一种决策标签，停止划分，返回这个决策标签
		if cateList.count(cateList[0]) == len(cateList): 
		    return cateList[0]
		# 程序终止条件2: 如果数据集的第一个决策标签只有一个 返回这个决策标签    
		if len(dataSet[0]) == 1:   	
		    return self.maxCate(cateList)		
		# 算法核心：
		bestFeat = self.getBestFeat(dataSet) # 返回数据集的最优特征轴：
		bestFeatLabel = labels[bestFeat]
		tree = {bestFeatLabel:{}}			
		del(labels[bestFeat])
		# 抽取最优特征轴的列向量
		uniqueVals = set([data[bestFeat] for data in dataSet]) # 去重	
		for value in uniqueVals:
		    subLabels = labels[:]  #将删除后的特征类别集建立子类别集
		    splitDataset = self.splitDataSet(dataSet, bestFeat, value) # 按最优特征列和值分割数据集
		    subTree = self.buildTree(splitDataset,subLabels) # 构建子树
		    tree[bestFeatLabel][value] = subTree
		return tree
	
	def maxCate(self,catelist):		# 计算出现最多的类别标签
		items = dict([(catelist.count(i), i) for i in catelist])
		return items[max(items.keys())]
			 	
	def getBestFeat(self,dataSet):
		# 计算特征向量维，其中最后一列用于类别标签，因此要减去 	
		numFeatures = len(dataSet[0]) - 1               # 特征向量维数= 行向量维度-1
		baseEntropy = self.computeEntropy(dataSet)      # 基础熵：源数据的香农熵
		bestInfoGain = 0.0;                             # 初始化最优的信息增益
		bestFeature = -1                                # 初始化最优的特征轴  		
		# 外循环：遍历数据集各列,计算最优特征轴
		# i 为数据集列索引：取值范围 0~(numFeatures-1)
		for i in xrange(numFeatures):		# 抽取第i列的列向量 			 	
			uniqueVals = set([data[i] for data in dataSet])	# 去重：该列的唯一值集	    
			newEntropy = 0.0                 # 初始化该列的香农熵		    
			for value in uniqueVals:         # 内循环：按列和唯一值计算香农熵				
				subDataSet = self.splitDataSet(dataSet, i, value) # 按选定列i和唯一值分隔数据集
				prob = len(subDataSet)/float(len(dataSet))
				newEntropy += prob * self.computeEntropy(subDataSet)          
			infoGain = baseEntropy - newEntropy   # 计算最大增益
			if (infoGain > bestInfoGain):         # 如果信息增益>0;    
				bestInfoGain = infoGain             # 用当前信息增益值替代之前的最优增益值 
				bestFeature = i                     # 重置最优特征为当前列
		return bestFeature 
		
	def computeEntropy(self,dataSet):                         # 计算香农熵
		datalen = float(len(dataSet))
		cateList = [data[-1] for data in dataSet]               # 从数据集中得到类别标签
		items = dict([(i,cateList.count(i)) for i in cateList]) # 得到类别为key，出现次数value的字典
		infoEntropy = 0.0                                       # 初始化香农熵    			
		for key in items:                                       # 计算香农熵
			prob = float(items[key])/datalen 
			infoEntropy -= prob * math.log(prob,2) # 香农熵：= - p*log2(p) --infoEntropy = -prob * log(prob,2)
		return infoEntropy	 
	# 分隔数据集：删除特征轴所在的数据列，返回剩余的数据集
	# dataSet：数据集;	 axis：特征轴;	 value：特征轴的取值
	def splitDataSet(self, dataSet, axis, value):		
		rtnList = []     
		for featVec in dataSet:
			if featVec[axis] == value:
				rFeatVec = featVec[:axis] # list操作 提取0~(axis-1)的元素 
				rFeatVec.extend(featVec[axis+1:]) # list操作 将特征轴（列）之后的元素加回
				rtnList.append(rFeatVec)   
		return rtnList 
	
	def predict(self,inputTree,featLabels,testVec):	# 分类器 
		root = inputTree.keys()[0]    # 树根节点
		secondDict = inputTree[root]  # value-子树结构或分类标签
		featIndex = featLabels.index(root)  # 根节点在分类标签集中的位置
		key = testVec[featIndex] # 测试集数组取值 
		valueOfFeat = secondDict[key] # 
		if isinstance(valueOfFeat, dict): 
		    classLabel = self.predict(valueOfFeat, featLabels, testVec) # 递归分类
		else: classLabel = valueOfFeat
		return classLabel
	
	# 存储树到文件
	def storeTree(self,inputTree,filename):
		fw = open(filename,'w')
		pickle.dump(inputTree,fw)
		fw.close()
	
	# 从文件抓取树    
	def grabTree(self,filename):
		fr = open(filename)
		return pickle.load(fr)	                   	