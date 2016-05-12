# -*- coding: utf-8 -*-
# Filename : 02kMeans1.py

import time
import numpy as np
from Recommand_Lib import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

k = 4
dataSet = file2matrix("testData/4k2_far.txt","\t")  
dataMat = mat(dataSet[:,1:]) # 转换为矩阵形式
	
kmeans = KMeans(init='k-means++', n_clusters=4)
kmeans.fit(dataMat)

# 输出生成的ClustDist：对应的聚类中心(列1),到聚类中心的距离(列2),行与dataSet一一对应
drawScatter(plt,dataMat,size=20,color='b',mrkr='.')
# 绘制聚类中心
drawScatter(plt,kmeans.cluster_centers_,size=60,color='red',mrkr='D')
plt.show() 
'''
colors = ['r', 'b', 'g','black']
for k , col in zip( range(k) , colors):
    members = (kmeans.labels_ == k )
    pl.plot( dataMat[members, 0] , dataMat[members,1] , 'w', markerfacecolor=col, marker='.')
    pl.plot(kmeans.cluster_centers_[k,0], kmeans.cluster_centers_[k,1], 'o', markerfacecolor=col,\
            markeredgecolor='k', markersize=10)
pl.show()
'''