# -*- encoding: utf-8 -*-
# Filename : adaboost2.py

'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *
import matplotlib.pyplot as plt 
import copy 

def loadDataSet(fileName):
		recordlist = []
		fp = open(fileName,"rb") 	
		content = fp.read()
		fp.close()
		rowlist = content.splitlines() 	
		recordlist=[map(eval, row.split("\t")) for row in rowlist if row.strip()]	
		dataSet = mat(recordlist)[:,:-1]
		labels =  mat(recordlist)[:,-1].T
		return dataSet,labels

# dataMat:数据集,
# Column: 第几列
# threshVal:阈值
# threshSymb:分类分隔符,lt,gt符号
def splitDataSet(dataMat,Column,threshVal,operator):
    retArray = ones((shape(dataMat)[0],1)) # 与数据集行数相同的全1向量
    if operator == 'lt': # '小于'
        retArray[dataMat[:,Column] <= threshVal] = -1.0 
    else:  # '大于'
        retArray[dataMat[:,Column] > threshVal] = -1.0
    return retArray # 预测结果

# 单层决策树生成函数: 以最小误差作为衡量标准找到最优列,不等于符号(大于,小于),阈值和重估的分类标签
# dataSet: 数据集
# labellist: 类别标签
# D: 列向量每个元素的平均权重:1/总元素数
def decisionTree(dataSet,labellist,D):
    dataMat = mat(dataSet); labelMat = mat(labellist).T
    m,n = shape(dataMat)# 数据集行、列数
    numSteps = 10.0; # 迭代步数 
    bestFeat = {}; # 最优项列 
    bestClass = mat(zeros((m,1))) # 最优预测分类
    minError = inf # 初始化最小误差为无穷大
    for i in xrange(n): # 按列迭代
        rangeMin = dataMat[:,i].min(); # 最小值
        rangeMax = dataMat[:,i].max(); # 最大值
        stepSize = (rangeMax-rangeMin)/numSteps # 步长 = (最大值-最小值)/步长数
        for j in xrange(-1,int(numSteps)+1):# 对每个步长数迭代: -1~(numSteps)            
            threshVal = (rangeMin + float(j) * stepSize) # 计算域值:(最小值+迭代步数*步长数)
            for operator in ['lt', 'gt']: # operator 操作符，取值为两个: lt小于,gt大于--分类分隔符
                # 调用 splitDataSet方法,小于,大于
                predictedVals = splitDataSet(dataMat,i,threshVal,operator) 
                errSet = mat(ones((m,1))) # 初始化误差集为一个全1向量                
                errSet[predictedVals == labelMat] = 0 # 误差集：列向量的预测值 == 类别标签则赋值为0
                weightedError = D.T*errSet  # 权重误差 = D*误差数组:权重误差是个标量
                if weightedError < minError: 
                    minError = weightedError # 更新最小误差为权重误差
                    bestClass = predictedVals.copy() # 最优预测类
                    bestFeat['dim'] = i # 最优列
                    bestFeat['thresh'] = threshVal # 最优阈值
                    bestFeat['oper'] = operator # 最优分隔符号(大于或小于号) 
    return bestFeat,minError,bestClass
# 基于单层分类器的ADABOOST训练过程
# 通过修改D的值调整弱分类器的权重
# dataSet:数据集
# labellist:分类标签
# numIt:迭代次数
def adaBoostTrain(dataSet,labellist,numIt=40):
    weakClassSet = [] # 初始化弱分类器
    m = shape(dataSet)[0]
    D = mat(ones((m,1))/m)   #初始化D为平均权重
    aggClassSet = mat(zeros((m,1)))
    for i in xrange(numIt):
        bestFeat,error,EstClass = decisionTree(dataSet,labellist,D)  
        alpha = float(0.5*log((1.0-error)/max(error,1e-16))) # alpha计算公式，1e-16避免除0
        bestFeat['alpha'] = alpha  
        weakClassSet.append(bestFeat)  # 以数组形式存储弱分类器
        # 算法核心：D--权重修改公式：D*exp((+-)alpha)/sum(D)（Logistic）
        # +-号取决于是否错分，+正确划分，-错误划分
        wtx = multiply(-1*alpha*mat(labellist).T,EstClass) # 矩阵对应元素相乘:multiply矩阵点积
        D = multiply(D,exp(wtx)) # 为下次迭代计算新的D
        D = D/D.sum()        
        aggClassSet += alpha*EstClass # 累计预测类：
        # 如果 x>0 sign(x)=1; x<0 sign(x)=-1
        # 计算所有分类器的训练误差--累计误差
        totalErr = multiply(sign(aggClassSet) != mat(labellist).T,ones((m,1)))
        errorRate = totalErr.sum()/m # 计算总误差率        
        if errorRate == 0.0: break # 如果为0，分类完毕 跳出循环
    return weakClassSet,aggClassSet

# Ada分类器
def adaClassify(datToClass,classdictList):
    dataMat = mat(datToClass)#do stuff similar to last aggClassSet in adaBoostTrainDS
    m = shape(dataMat)[0]
    aggClassSet = mat(zeros((m,1)))
    for i in range(len(classdictList)):
        EstClass = splitDataSet(dataMat,classdictList[i]['dim'],classdictList[i]['thresh'],classdictList[i]['oper'])
        aggClassSet += classdictList[i]['alpha']*EstClass
    return sign(aggClassSet)

def plotROC(predStrengths, labellist):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(labellist)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(labellist)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if labellist[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep
