# -*- coding: utf-8 -*-
# Filename : trees2.py

'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

# 创建数据集
def createDataSet():
	  # 无需浮出水面,脚蹼,是否是鱼类
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers'] #[无需浮出水面,脚蹼]
    #change to discrete values
    return dataSet, labels

# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 得到数据集行数
    labelCounts = {}          # 初始化类别标签
    # featVec是指特征向量
    # 这段代码计算了数据集中各个特征向量的和
    for featVec in dataSet: 
    	  # featVec[-1]：数据集行中最后一个元素-特征向量：这里是yes no
        currentLabel = featVec[-1]
        # 如果当前的字典labelCounts没有currentLabel对应特征向量的键，在字典中加入新的特征向量这个键
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        # 在字典labelCounts中currentLabel对应特征向量的键值+1
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 初始化香农熵    
    # 计算香农熵
    for key in labelCounts:
    	# 计算各个特征向量的概率：特征向量出现的次数/总记录数
        prob = float(labelCounts[key])/numEntries
    # 香农熵：= - p*log2(p) --shannonEnt2 = -prob * log(prob,2)
    # 这里计算的是整个数据集累计的香农熵 
        shannonEnt -= prob * log(prob,2) 
    return shannonEnt
    
# 分隔数据集：删除特征轴所在的数据列，返回剩余的数据集
# dataSet：数据集
# axis：特征轴
# value：特征轴的取值
def splitDataSet(dataSet, axis, value):
	# 初始化划分后的数据集:list
    retDataSet = []     
    # 遍历数据集中所有行
    for featVec in dataSet:
    	# 如果featVec[axis]取值等于value
    	if featVec[axis] == value:
    		    # 从数据集中删除掉特征轴所在列
            reducedFeatVec = featVec[:axis] # list操作 提取0~(axis-1)的元素 
            # print "reducedFeatVec1:",reducedFeatVec
            reducedFeatVec.extend(featVec[axis+1:]) # list操作 将特征轴（列）之后的元素加回
            # print "reducedFeatVec2:",reducedFeatVec	
            # 把删除特征轴的划分数据附加到返回矩阵中 	
            retDataSet.append(reducedFeatVec) 
            # print "retDataSet:",retDataSet
    # 返回划分后的特征矩阵  
    return retDataSet
    
# 从数据集中选择最优的特征        
def chooseBestFeatureToSplit(dataSet):
    # 计算特征向量维，其中最后一列用于类别标签，因此要减去 	
    numFeatures = len(dataSet[0]) - 1          # 特征向量维数= 行向量维度-1
    baseEntropy = calcShannonEnt(dataSet)      # 基础熵：源数据的香农熵
    # print "baseEntropy:",baseEntropy
    bestInfoGain = 0.0;    # 初始化最优的信息增益
    bestFeature = -1       # 初始化最优的特征轴
    
    # 外循环：遍历数据集各列,计算最优特征轴
    # i 为数据集列索引：取值范围 0~(numFeatures-1)
    for i in range(numFeatures):

	      # 抽取第i列的列向量  	
        featList = [example[i] for example in dataSet]
        
        uniqueVals = set(featList)     # 去重：该列的唯一值集
        newEntropy = 0.0               # 初始化该列的香农熵 
        
        # 内循环：按列和唯一值计算香农熵
        for value in uniqueVals:
        	  # 按选定列i和唯一值分隔数据集--删除选定列，返回剩余的数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # print "subDataSet:",subDataSet
            # 概率：prob= 子数据集的行数/源数据集的行数
            prob = len(subDataSet)/float(len(dataSet))
            # 新香农熵：子数据集的概率*子数据集的香农熵
            # print "prob * calcShannonEnt(subDataSet):",prob * calcShannonEnt(subDataSet)
            # 累计新香农熵：newEntropy 
            newEntropy += prob * calcShannonEnt(subDataSet)  
        # print "newEntropy:",newEntropy  
        # 根据新香农熵与基础熵比较计算信息量的增益（本质是熵的减少，无序度的减少）        
        infoGain = baseEntropy - newEntropy   
        if (infoGain > bestInfoGain):       # 如果信息增益>0;    
            bestInfoGain = infoGain         # 用当前信息增益值替代之前的最优增益值 
            bestFeature = i                 # 重置最优特征为当前列
    return bestFeature                      

#计算最多的类别标签
def majorityCnt(classList):
	  # 初始化字典：
    # key: 类别标签；value: 数量
    classCount={}   
	  # 迭代将classList的向量值赋予classCount中
    for vote in classList:
   	    # 如果vote不存在在字典classCount的键中, 就加入这个键
        if vote not in classCount.keys(): classCount[vote] = 0
        # 对应vote的字典值+1        	
        classCount[vote] += 1
    # 对分类数按value重新排序
    # 该句是按字典值排序的固定用法
    # classCount.iteritems()：字典迭代器函数
    # key：排序参数 operator.itemgetter(1)：多级排序         
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # print "sortedClassCount:",sortedClassCount
    # 返回出现最多类别标签
    return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet,labels):	  
    classList = [example[-1] for example in dataSet] # 抽取源数据集的决策标签列
    # 程序终止条件1
    # 统计第一个标签的数量：classList.count(classList[0])
    # 如果classList只有一种决策标签，停止划分，返回这个决策标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    # 程序终止条件2
    # 如果数据集的第一个决策标签只有一个
    # 返回最多的决策标签    
    if len(dataSet[0]) == 1:   	
        return majorityCnt(classList)
    
    # 算法核心：    
    # 返回数据集的最优特征轴：这个特征轴的香农熵<数据集的香农熵    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取最优的特征标签用于创建树
    bestFeatLabel = labels[bestFeat]
    # 构建决策树,树的结构：广义表的形式 
    # key:最优特征轴标签; value: subTree
    myTree = {bestFeatLabel:{}}
    	
    # 删除labels数组中对应的特征类别--即表示已经处理完成	
    del(labels[bestFeat])
    
    # 抽取最优特征轴的列向量
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)# 去重    
    
    for value in uniqueVals:
        subLabels = labels[:]  #将删除后的特征类别集建立子类别集
        # print subLabels	
        # 按最优特征轴和唯一值分隔数据集--删除特征轴的数据列，返回剩余的数据集
        splitDataset = splitDataSet(dataSet, bestFeat, value)
        # print splitDataset 
        # 对分隔后的数据集按照子特征类别集递归--树的生长
        # 子树的数据结构: 键:唯一值; 值:类别标签或子树(递归返回)
        subTree = createTree(splitDataset,subLabels)
        # print "bestFeatLabel:",bestFeatLabel,"value:",value,"subTree:",subTree
        myTree[bestFeatLabel][value] = subTree
        
        # 对分隔后的数据集进行按照子特征类别集进行递归	
        # myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
        
    return myTree  # 返回生成后的决策树                          

# 分类器    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]    # 树根节点
    secondDict = inputTree[firstStr]  # value-子树结构或分类标签
    featIndex = featLabels.index(firstStr)  # 根节点在分类标签集中的位置
    key = testVec[featIndex] # 测试集数组取值 
    valueOfFeat = secondDict[key] # 
    # 判断 valueOfFeat 是否是 dict类型
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec) # 递归分类
    else: classLabel = valueOfFeat
    return classLabel

# 存储树到文件
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

# 从文件抓取树    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
