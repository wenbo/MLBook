# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import sys,os
import copy
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt  

class Eigenfaces(object):
	def __init__(self):
		self.eps = 1.0e-16
		self.X = []
		self.y = []
		self.Mat=[]
		self.eig_v = 0
		self.eig_vect = 0
		self.mu = 0
		self.projections = []
		self.dist_metric=0
	def loadimgs(self,path): # 加载图片数据集
		classlabel = 0
		for dirname, dirnames, filenames in os.walk(path):
			for subdirname in dirnames:
				sub_path = os.path.join(dirname, subdirname)
				for filename in os.listdir(sub_path):
					im = Image.open(os.path.join(sub_path, filename))
					im = im.convert("L") #数据转换为long类型
					self.X.append(np.asarray(im, dtype=np.uint8))
					self.y.append(classlabel)
				classlabel += 1	
	# 将图片变为行向量	# 生成图片矩阵
	def genRowMatrix(self):
		self.Mat = np.empty((0, self.X[0].size), dtype=self.X[0].dtype)
		for row in self.X:
			self.Mat = np.vstack((self.Mat, np.asarray(row).reshape(1,-1)))
	# 计算特征脸
	def PCA(self, pc_num =0):
		self.genRowMatrix()	
		[n,d] = shape(self.Mat)
		if ( pc_num <= 0) or ( pc_num>n):		pc_num = n
		self.mu = self.Mat.mean(axis =0)
		self.Mat -= self.mu
		if n>d:
			XTX = np.dot (self.Mat.T,self.Mat)
			[ self.eig_v , self.eig_vect ] = linalg.eigh (XTX)
		else :
			XTX = np.dot(self.Mat,self.Mat.T)
			[ self.eig_v , self.eig_vect ] = linalg.eigh (XTX)
		self.eig_vect = np.dot(self.Mat.T, self.eig_vect)
		for i in xrange(n):
			self.eig_vect[:,i] = self.eig_vect[:,i]/linalg.norm(self.eig_vect[:,i])
		idx = np.argsort(-self.eig_v)
		self.eig_v = self.eig_v[idx]
		self.eig_vect = self.eig_vect[:,idx ]		
		self.eig_v = self.eig_v[0:pc_num ].copy () # select only pc_num
		self.eig_vect = self.eig_vect[:,0:pc_num].copy ()
			 
	def compute(self):
		self.PCA()
		for xi in self.X:
			self.projections.append(self.project(xi.reshape(1,-1))) 
	
	def distEclud(self, vecA, vecB):  # 欧氏距离
		return linalg.norm(vecA-vecB)+self.eps 
	
	def cosSim(self, vecA, vecB):	 # 夹角余弦	
		return (dot(vecA,vecB.T)/((linalg.norm(vecA)*linalg.norm(vecB))+self.eps))[0,0]
	# 映射
	def project(self,XI):
		if self.mu is None:	return np.dot(XI,self.eig_vect)
		return np.dot(XI-self.mu, self.eig_vect)	
	#预测最接近的特征脸
	def predict(self,XI):
		minDist = np.finfo('float').max
		minClass = -1
		Q = self.project(XI.reshape(1,-1))
		for i in xrange(len(self.projections)):
			dist = self.dist_metric(self.projections[i], Q)
			if dist < minDist:
				minDist = dist
				minClass = self.y[i]
		return minClass
	# 生成特征脸
	def subplot(self,title, images):
		fig = plt.figure()
		fig.text(.5, .95, title, horizontalalignment='center') 
		for i in xrange(len(images)):
			ax0 = fig.add_subplot(4,4,(i+1))
			plt.imshow(asarray(images[i]), cmap="gray")
			plt.xticks([]), plt. yticks([]) # 隐藏 X Y 坐标
		plt.show()
	# 归一化
	def normalize(self, X, low, high, dtype=None):
		X = np.asarray(X)
		minX, maxX = np.min(X), np.max(X)
		X = X - float(minX)
		X = X / float((maxX - minX))
		X = X * (high-low)
		X = X + low
		if dtype is None:
			return np.asarray(X)
		return np.asarray(X, dtype=dtype)
'''		
	# 重构
	def reconstruct(self,W, Y, mu=None):
		if mu is None:	return np.dot(Y,W.T)
		return np.dot(Y, W.T) + mu
	# 从外部数据计算投影
	def out_project(self,W,XI,mu):
		if mu is None:	return np.dot(XI,W)
		return np.dot(XI-mu, W)	
'''