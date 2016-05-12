# -*- coding: utf-8 -*-

from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

# 二元切分数据集
def binSplitDataSet(dataSet, feature, value):	
    # nonzero(dataSet[:,feature] > value)[0]: 数据集第feature列大于value(特征值)的行向量
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    # nonzero(dataSet[:,feature] <= value)[0]: 数据集第feature列小于等于value(特征值)的行向量	
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

# 回归树叶子节点
def regLeaf(dataSet):
    return mean(dataSet[:,-1]) # 返被回划分数据集最后一列的均值

# 回归方差
# 返回数据集最后1列的二阶中心距乘以划分数据集的行数
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0] 

# 选择最优分割点
# leafType:叶子节点算法函数
# errType:回归方差算法函数
# ops:允许的方差下降值,最小切分样本数
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):    
    tolS = ops[0]; # 允许的方差下降值
    tolN = ops[1] # 最小切分样本数    
    #---- 算法终止条件1开始 ----#
    splitdataSet = set(dataSet[:,-1].T.tolist()[0])
    if len(splitdataSet) == 1:   
        return None, leafType(dataSet)   # 返回值: leafType(dataSet):树的叶子节点    
    #---- 算法终止条件1结束 ----#
    
    #---- 计算dataSet各列的最优划分方差,划分列,划分值 ----#
    m,n = shape(dataSet) # 返回数据集的行数和列数
    S = errType(dataSet)    # 计算整个数据集的回归方差,S
    # 初始化最优参数: 最大方差、最优划分列、最优划分值
    bestS = inf; bestIndex = 0; bestValue = 0
    # 按列遍历数据集前n-1列
    # featIndex: 第0~n-1列
    for featIndex in xrange(n-1):
        # 遍历每列去重后的各个类型
        # splitVal:各列的每个类型
        for splitVal in set(dataSet[:,featIndex]):
            # 二元划分数据集:按划分列和划分值分隔dataSet
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # mat0的行数 小于 tolN 或 mat1的行数 小于 tolN
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue # 终止后面的程序,进行下一次循环
            # mat0的回归方差+mat1的回归方差	
            newS = errType(mat0) + errType(mat1)
            # 如果newS小于bestS
            if newS < bestS: 
                bestIndex = featIndex # 最优索引 <- 特征索引
                bestValue = splitVal # 最优值 <- 分割值
                bestS = newS # bestS <- newS            
    #---- DataSet的最优划分参数：方差、划分列、划分值计算结束 ----#
    
    #---- 算法终止条件2开始:返回的是值节点类型 ----#  
    # 如果(S - bestS) 小于 1
    if (S - bestS) < tolS: 
        # print "stop 2"
        # 返回值: feat==None,leafType(dataSet):数据集标签集的均值   	
        return None, leafType(dataSet) 
    #---- 算法终止条件2结束 ----#
    
    #---- 算法终止条件3开始 ----#        
    # 二元划分数据集:按划分列和划分值分隔dataSet
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # mat0的行数小于 tolN 或 mat1的行数小于tolN 
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):       	 
    	# print "stop 3"
    	return None, leafType(dataSet) 
    #---- 算法终止条件3结束 ----#  
    # 算法终止的前3个条件的划分列为None,说明为叶子节点,本枝分类树划分结束 
    
    #---- 算法终止条件4开始:返回的是子树节点类型 ----#
    # print "stop 4"  
    # 返回最优特征的划分列和划分值,但回归树还需递归划分   
    return bestIndex,bestValue 
    #---- 算法终止条件4结束 ----#                          

# 创建分类回归树
# dataSet: 数据集矩阵
# leafType:叶子节点算法函数指针
# errType:回归方差算法函数指针
# ops: 允许的方差下降值,最小切分样本数--算法停止条件
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 选择最优划分: feat: 划分列, val: 划分值
    # 传递 叶子节点算法函数指针,回归方差算法函数指针,允许的方差下降值,最小切分样本数
    feat,val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 停止条件:如果feat为空,算法停止, 返回叶子节点值
    if feat == None:  	return val 
    retTree = {}
    retTree['spInd'] = feat # 把划分列特征放入创建的树中   
    retTree['spVal'] = val # 把划分值放入创建的树中 
    # 因为是面向过程的编程，判断树的节点级输出有些麻烦,增加调试信息：
    # 输出节点一级的信息,连续输出多少个就有多少层
    # 本算法为先左后右的递归，所以首先输出的是left node，当node输出中断后，才为另一侧的node
    # print "node:",retTree  
    # 以划分列和元素为分割点二分数据集:dataSet被分为左右两部分：lSet:左子树集合, rSet:右子树集合
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归生成子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree  

# 判断是否为树,验证输入数据是否为字典
def isTree(obj):
    return (type(obj).__name__=='dict')

# 计算树叶子节点的均值
def getMean(tree):
    # 左、右子树递归至叶子节点处		
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    # 返回叶子节点的均值    	
    return (tree['left']+tree['right'])/2.0

# 树的后剪枝
# testData: 测试集    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) # 如果没有测试数据输入,运行getMean,程序退出
    # 如果左、右子节点是树    	
    if (isTree(tree['right']) or isTree(tree['left'])):
        # 对测试集按划分列和树的划分值进行二元分割    	        
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet) # 如果左节点是树，对测试集递归剪枝
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet) # 如果右节点是树，对测试集递归剪枝 
    # 如果左右节点都不是树
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal']) # 对测试集按划分列和划分值进行二元分割
        # 计算左右子树的方差
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))
        # 执行合并：树节点均值
        treeMean = (tree['left']+tree['right'])/2.0
        # 计算合并的方差
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        # 如果合并后的方差小于不合并方差
        if errorMerge < errorNoMerge: 
            # print "merging"
            return treeMean  # 返回节点均值,执行合并
        else: return tree # 否则直接返回,不进行合并
    else: return tree

# 将模型数据格式化为线性函数的自变量X,因变量Y
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1))) #初始化全X,Y
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1] # 为 X,Y 赋值
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y) # 产生截距和斜率矩阵
    return ws,X,Y # 返回X,Y和线性回归的系数
    
# 产生叶子节点的模型，并返回系数
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

# 返回线性模型的预测值与实际值的方差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

# 回归树评估: 返回树节点的浮点值     
def regTreeEval(model, inDat):
    return float(model)

# 模型树评估:Y = X*model
# model为二维向量: 斜率,截距
def modelTreeEval(model, inDat):
    n = shape(inDat)[1] # inDat的列数
    X = mat(ones((1,n+1))) # 全1矩阵,1行,n+1列
    X[:,1:n+1] = inDat 
    return float(X*model) # Y = X*model

# 树预测
# tree:回归树，模型树
# inData:第i行的测试集向量 
# modelEval:与树一致的评估函数
def treeForeCast(tree, inData, modelEval=regTreeEval):
    # 如果不是树直接返回评估结果
    if not isTree(tree): return modelEval(tree, inData)
    # 如果inData[划分列]的取值大于tree[划分值]
    if inData[tree['spInd']] > tree['spVal']:
        # 如果tree的左子树是一颗树,而不是叶子节点 递归调用本函数
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        # 否则返回叶子节点对应评估函数的评估值
        else: return modelEval(tree['left'], inData)
    else: # 如果inData[划分列]的取值小于等于tree[划分值]
        # 如果tree的右子树是一颗树,而不是叶子节点,递归调用本函数        
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        # 否则返回叶子节点对应评估函数的评估值        
        else: return modelEval(tree['right'], inData)

# 创建预测
# tree: 回归树，模型树 
# testData: 测试集 
# modelEval: 评估函数 regTreeEval,modelTreeEval      
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1))) # 初始化为全0向量,行数为m
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat