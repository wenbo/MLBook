# -*- coding: utf-8 -*-            
# Filename : 03svmRbf.py        
                                   
from numpy import *                
import numpy as np                 
import operator                    
import svmMLiA2                    
import matplotlib.pyplot as plt    

k1=1.3
# 加载训练集
dataArr,labelArr = svmMLiA2.loadDataSet('testSetRBF.txt')
# 使用Platt SMO分类
# 使用rbf非线性核函数
b,alphas = svmMLiA2.smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
datMat=mat(dataArr); labelMat = mat(labelArr).T
svInd=nonzero(alphas.A>0)[0]
# 获取支持向量
sVs=datMat[svInd] #get matrix of only support vectors
labelSV = labelMat[svInd];
# 输出支持向量的数量
print "there are %d Support Vectors" % shape(sVs)[0]
print svInd
m,n = shape(datMat)
errorCount = 0
# 计算训练错误率
for i in range(m):
    kernelEval = svmMLiA2.kernelTrans(sVs,datMat[i,:],('rbf', k1))
    predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
    if sign(predict)!=sign(labelArr[i]): errorCount += 1
print "the training error rate is: %f" % (float(errorCount)/m)

# 输出非线性分类图
mydata = mat(dataArr)
# 数据描点
fig = plt.figure()
ax = fig.add_subplot(111) 
for i in range(len(mydata)):
	if alphas[i]!=0: # KKT条件
		ax.scatter(mydata[i,0],mydata[i,1],c='green',marker='s')		
	elif labelArr[i] == 1:
		ax.scatter(mydata[i,0],mydata[i,1],c='blue',marker='o')
	elif labelArr[i] == -1:
		ax.scatter(mydata[i,0],mydata[i,1],c='red',marker='o')
# 显示绘制的图形
plt.show()
# 加载测试集
dataArr,labelArr = svmMLiA2.loadDataSet('testSetRBF2.txt')
errorCount = 0
datMat=mat(dataArr); labelMat = mat(labelArr).T
m,n = shape(datMat)
# 用核函数划分测试集
for i in range(m):
    kernelEval = svmMLiA2.kernelTrans(sVs,datMat[i,:],('rbf', k1))
    predict= kernelEval.T * multiply(labelSV,alphas[svInd]) + b
    if sign(predict)!=sign(labelArr[i]): errorCount += 1    
# 输出误差分类结果    	
print "the test error rate is: %f" % (float(errorCount)/m)   