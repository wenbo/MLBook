# -*- coding: utf-8 -*-

from numpy import *
import math
import copy
import cPickle as pickle

class C45DTree(object):
	def __init__(self):
		self.tree={}
		self.dataSet=[]
		self.labels=[]
	
	def loadDataSet(self,path,labels):
		recordlist = []
		fp = open(path,"rb") 	
		content = fp.read()
		fp.close()
		rowlist = content.splitlines() 	
		recordlist=[row.split("\t") for row in rowlist if row.strip()]	
		self.dataSet = recordlist
		self.labels = labels
		
	def train(self):
		labels = copy.deepcopy(self.labels)
		self.tree = self.buildTree(self.dataSet,labels)

	def buildTree(self,dataSet,labels):	  
		cateList = [data[-1] for data in dataSet]
		if cateList.count(cateList[0]) == len(cateList): 
		    return cateList[0]
		if len(dataSet[0]) == 1:   	
		    return self.maxCate(cateList)		
		bestFeat, featValueList = self.getBestFeat(dataSet)
		bestFeatLabel = labels[bestFeat]
		tree = {bestFeatLabel:{}}			
		del(labels[bestFeat])		
		for value in featValueList:
		    subLabels = labels[:] 
		    splitDataset = self.splitDataSet(dataSet, bestFeat, value) 
		    subTree = self.buildTree(splitDataset,subLabels) 
		    tree[bestFeatLabel][value] = subTree
		return tree
	
	def maxCate(self,catelist):	
		items = dict([(catelist.count(i), i) for i in catelist])
		return items[max(items.keys())]
		
	def getBestFeat(self, dataSet):
		Num_Feats = len(dataSet[0][:-1])
		totality = len(dataSet)
		BaseEntropy = self.computeEntropy(dataSet)
		ConditionEntropy = [] # 初始化条件熵
		slpitInfo = []  # for C4.5, calculate gain ratio
		allFeatVList=[]
		for f in xrange(Num_Feats):
			featList = [example[f] for example in dataSet]
			[splitI,featureValueList] = self.computeSplitInfo(featList)
			allFeatVList.append(featureValueList)         
			slpitInfo.append(splitI)		    
			resultGain = 0.0		    
			for value in featureValueList:
				subSet = self.splitDataSet(dataSet, f, value)
				appearNum = float(len(subSet))
				subEntropy = self.computeEntropy(subSet)
				resultGain += (appearNum/totality)*subEntropy
			ConditionEntropy.append(resultGain) # 总条件熵
		infoGainArray = BaseEntropy*ones(Num_Feats)-array(ConditionEntropy)
		infoGainRatio = infoGainArray/array(slpitInfo) # c4.5, info gain ratio
		bestFeatureIndex =  argsort(-infoGainRatio)[0]
		return bestFeatureIndex, allFeatVList[bestFeatureIndex]		
		
	def computeSplitInfo(self, featureVList):
		numEntries = len(featureVList)
		featureVauleSetList = list(set(featureVList))
		valueCounts = [featureVList.count(featVec) for featVec in featureVauleSetList]
		# caclulate shannonEnt
		pList = [float(item)/numEntries for item in valueCounts ]
		lList = [item*math.log(item,2) for item in pList]
		splitInfo = -sum(lList)
		return splitInfo, featureVauleSetList		

	def computeEntropy(self,dataSet):  
		datalen = float(len(dataSet))
		cateList = [data[-1] for data in dataSet] 
		items = dict([(i,cateList.count(i)) for i in cateList]) 
		infoEntropy = 0.0 
		for key in items: 
			prob = float(items[key])/datalen 
			infoEntropy -= prob * math.log(prob,2) 
		return infoEntropy	 

	def splitDataSet(self, dataSet, axis, value):		
		rtnList = []     
		for featVec in dataSet:
			if featVec[axis] == value:
				rFeatVec = featVec[:axis] 
				rFeatVec.extend(featVec[axis+1:]) 
				rtnList.append(rFeatVec)   
		return rtnList 
	
	# 树的后剪枝
# testData: 测试集    
def prune(tree, testData):
	pass
	

	def predict(self,inputTree,featLabels,testVec):	
		root = inputTree.keys()[0]    
		secondDict = inputTree[root] 
		featIndex = featLabels.index(root) 
		key = testVec[featIndex] 
		valueOfFeat = secondDict[key] # 
		if isinstance(valueOfFeat, dict): 
		    classLabel = self.predict(valueOfFeat, featLabels, testVec)
		else: classLabel = valueOfFeat
		return classLabel

	def storeTree(self,inputTree,filename):
		fw = open(filename,'w')
		pickle.dump(inputTree,fw)
		fw.close()
 
	def grabTree(self,filename):
		fr = open(filename)
		return pickle.load(fr)	                   	