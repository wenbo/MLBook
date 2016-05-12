# -*- coding: utf-8 -*-
# Filename : Recommand_lib.py

from numpy import *
import numpy as np 
import operator
import scipy.spatial.distance as dist

# 数据文件转矩阵
# path: 数据文件路径
# delimiter: 文件分隔符
def file2matrix(path,delimiter):	
	recordlist = []
	fp = open(path,"rb") 	# 读取文件内容
	content = fp.read()
	fp.close()
	rowlist = content.splitlines() 	# 按行转换为一维表
	# 逐行遍历 		# 结果按分隔符分割为行向量
	recordlist=[map(eval, row.split(delimiter)) for row in rowlist if row.strip()]	
	return mat(recordlist)	# 返回转换后的矩阵形式

# 随机生成聚类中心	
def randCenters(dataSet, k):
    n = shape(dataSet)[1]
    clustercents = mat(zeros((k,n)))# 初始化聚类中心矩阵:k*n 
    for col in xrange(n):
        mincol = min(dataSet[:,col]); maxcol = max(dataSet[:,col])
        # random.rand(k,1): 产生一个0~1之间的随机数向量：k,1表示产生k行1列的随机数
        clustercents[:,col] = mat(mincol + float(maxcol - mincol) * random.rand(k,1))
    return clustercents
# 欧氏距离
eps = 1.0e-6
def distEclud(vecA, vecB):
	return linalg.norm(vecA-vecB)+eps 
# 相关系数
def distCorrcoef(vecA, vecB):
	return corrcoef(vecA, vecB, rowvar = 0)[0][1]
# Jaccard距离	
def distJaccard(vecA, vecB):
	temp = mat([array(vecA.tolist()[0]),array(vecB.tolist()[0])])
	return dist.pdist(temp,'jaccard')
# 余弦相似度
def cosSim(vecA, vecB):	
	return (dot(vecA,vecB.T)/((linalg.norm(vecA)*linalg.norm(vecB))+eps))[0,0]
# 绘制散点图    
def drawScatter(plt,mydata,size=20,color='blue',mrkr='o'):
	plt.scatter(mydata.T[0],mydata.T[1],s=size,c=color,marker=mrkr)   

# 根据聚类范围绘制散点图
def color_cluster(dataindx,dataSet,plt,k=4):
	index = 0
	datalen = len(dataindx)
	for indx in xrange(datalen):
		if int(dataindx[indx]) ==0:
			plt.scatter(dataSet[index,0],dataSet[index,1],c='blue',marker='o')
		elif int(dataindx[indx]) ==1:
			plt.scatter(dataSet[index,0],dataSet[index,1],c='green',marker='o')
		elif int(dataindx[indx]) ==2:
			plt.scatter(dataSet[index,0],dataSet[index,1],c='red',marker='o')
		elif int(dataindx[indx]) ==3:
			plt.scatter(dataSet[index,0],dataSet[index,1],c='cyan',marker='o')
		index += 1				

# KMeans 主函数
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCenters):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            distlist =[ distMeas(centroids[j,:],dataSet[i,:]) for j in range(k) ]
            minDist = min(distlist)
            minIndex = distlist.index(minDist)            	
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0) 
    return centroids, clusterAssment
    
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss #reassign new clusters, and SSE
    return mat(centList), clusterAssment

# dataSet 训练集
# testVect 测试集
# r=3 取前r个近似值
# rank=1,结果排序
# distCalc 相似度计算函数
def recommand(dataSet,testVect,r=3,rank=1,distCalc=cosSim):
	m,n = shape(dataSet)
	limit = min(m,n)
	if r>limit: r=limit
	U,S,VT = linalg.svd(dataSet.T) # svd分解
	V =VT.T
	Ur = U[:,:r]   # 取前r个U,S,V值
	Sr = diag(S)[:r,:r]
	Vr = V[:,:r]
	testresult = testVect*Ur*linalg.inv(Sr)  # 计算User E的坐标值	
	# 计算测试集与训练集每个记录的相似度
	resultarray = array([ distCalc(testresult,vi) for vi in Vr ])  
	descindx = argsort(-resultarray)[:rank] # 排序结果--降序
	return  descindx ,resultarray[descindx] # 排序后的索引和值
	
def kNN(testdata, trainSet, listClasses, k):
    dataSetSize = trainSet.shape[0]    
    distances = array(zeros(dataSetSize))
    for indx in xrange(dataSetSize):
    	distances[indx] = cosSim(testdata,trainSet[indx])
    sortedDistIndicies = argsort(-distances)  
    classCount={}            
    for i in range(k):  # i = 0~(k-1)  	  
        voteIlabel = listClasses[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]	