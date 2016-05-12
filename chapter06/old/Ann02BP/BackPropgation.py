# -*- coding: UTF-8 -*-
# Filename : BackPropgation.py
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: jack zheng
'''
from numpy import *
import operator
import Untils
import matplotlib.pyplot as plt 

# 传递函数:
def logistic(inX):
    return 1.0/(1+exp(-inX))

# 传递函数的导函数
def dlogit(inX1,inX2):
    return multiply(inX2,(1-inX2))

# 矩阵各元素平方之和
def sumsqr(inX):
    return sum(power(inX,2))
    
# 加载student.txt数据集
def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename) #testSet.txt
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1]), 1.0])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat   

# 数据标准化(归一化):student.txt数据集
def normalize(dataMat):
    # 计算均值
    height = mean(dataMat[:,0])
    weight = mean(dataMat[:,1])	 
    # 计算均方差
    stdh = std(dataMat[:,0])
    stdw = std(dataMat[:,1])
    # 标准化
    dataMat[:,0] = (dataMat[:,0]-height)/stdh
    dataMat[:,1] = (dataMat[:,1]-weight)/stdw	 
    return dataMat

def bpNet(dataSet,classLabels):
    # 数据集矩阵化
    SampIn = mat(dataSet).T
    expected = mat(classLabels)
    [m,n] = shape(dataSet) 
    # 网络参数
    eb = 0.01                   # 误差容限 
    eta = 0.05                   # 学习率 
    mc = 0.2                    # 动量因子 
    maxiter = 2000              # 最大迭代次数 
    errRec = []                 # 误差
    # 构造网络
    
    # 初始化网络
    nSampNum = m;  # 样本数量
    nSampDim = n-1;  # 样本维度
    nHidden = 4;   # 隐含层神经元 
    nOut = 1;      # 输出层

    # 输入层参数
    
    # 隐含层参数
    # net_Hidden * 3 一行代表一个隐含层节点
    w = 2.0*(random.rand(nHidden,nSampDim)-1.0/2.0)  
    b = 2.0*(random.rand(nHidden,1)-1.0/2.0) 
    wex = mat(Untils.mergMatrix(mat(w),mat(b)))
    
    # 输出层参数
    W = 2.0*(random.rand(nOut,nHidden)-1.0/2.0) 
    B = 2.0*(random.rand(nOut,1)-1.0/2.0) 
    WEX = mat(Untils.mergMatrix(mat(W),mat(B)))
    
    dWEXOld = 0.0 ; dwexOld = 0.0 
    # 训练
    iteration = 0.0;  
    for i in range(maxiter):   
        # 1. 工作信号正向传播
        hp = wex*SampIn
        tau = logistic(hp)
        tauex  = Untils.mergMatrix(tau.T, ones((nSampNum,1))).T
    
        HM = WEX*tauex
        out = logistic(HM)    
        err = expected - out 
        sse = sumsqr(err) 
        errRec.append(sse); 
        # 判断是否收敛
        iteration = iteration + 1    
        if sse <= eb:
            print "iteration:",i    
            break;
         
        # 2.误差信号反向传播
        # DELTA和delta为局部梯度  
        DELTA = multiply(err,dlogit(HM,out))
        wDelta = W.T*DELTA
        delta = multiply(wDelta,dlogit(hp,tau))
        dWEX = DELTA*tauex.T 
        dwex = delta*SampIn.T        
        
        # 3.更新权值
        if i == 0:  
            WEX = WEX + eta * dWEX
            wex = wex + eta * dwex
        else :    
            WEX = WEX + (1.0 - mc)*eta*dWEX + mc * dWEXOld
            wex = wex + (1.0 - mc)*eta*dwex + mc * dwexOld
     
        dWEXOld = dWEX
        dwexOld = dwex 
        W  = WEX[:,0:nHidden]
    return errRec,WEX,wex 

def BPClassfier(start,end,WEX,wex):
    x = linspace(start,end,30)
    xx = mat(ones((30,30)))
    xx[:,0:30] = x 
    yy = xx.T
    z = ones((len(xx),len(yy))) ;
    for i in range(len(xx)):
    	for j in range(len(yy)):
         xi = []; tauex=[] ; tautemp=[]
         mat(xi.append([xx[i,j],yy[i,j],1])) 
         hp = wex*(mat(xi).T)
         tau = logistic(hp) 
         taumrow,taucol= shape(tau)
         tauex = mat(ones((1,taumrow+1)))
         tauex[:,0:taumrow] = (tau.T)[:,0:taumrow]
         HM = WEX*(mat(tauex).transpose())
         out = logistic(HM) 
         z[i,j] = out 
    return x,z             