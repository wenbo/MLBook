# -*- coding: UTF-8 -*-
# Filename : BackPropgation.py
'''
Created on Oct 27, 2010
BP Working Module
@author: jack zheng
'''
from numpy import *
import operator
import Untils
import matplotlib.pyplot as plt 

# 传递函数:
def logistic(inX):
    return 1.0/(1.0+exp(-inX))

# 传递函数的导函数
def dlogit(inX1,inX2):
    return multiply(inX2,(1.0-inX2))

# 矩阵各元素平方之和
def errorfunc(inX):
    return sum(power(inX,2))/2.0
    
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
    # 标准化
    dataMat[:,0] = (dataMat[:,0]-mean(dataMat[:,0]))/std(dataMat[:,0])
    dataMat[:,1] = (dataMat[:,1]-mean(dataMat[:,1]))/std(dataMat[:,1])
    return dataMat

def bpNet(dataSet,classLabels):
    # 数据集矩阵化
    SampIn = mat(dataSet).T
    expected = mat(classLabels)
    m,n = shape(dataSet) 
    # 网络参数
    eb = 0.01                   # 误差容限 
    eta = 0.05                  # 学习率 
    mc = 0.3                    # 动量因子 
    maxiter = 2000              # 最大迭代次数 
    errlist = []                # 误差列表
    
    # 构造网络    
    # 初始化网络
    nSampNum = m;    # 样本数量
    nSampDim = n-1;  # 样本维度
    nHidden = 4;   # 隐含层神经元 
    nOut = 1;      # 输出层
    
    # 隐含层参数
    hi_w = 2.0*(random.rand(nHidden,nSampDim)-0.5)  
    hi_b = 2.0*(random.rand(nHidden,1)-0.5) 
    hi_wb = mat(Untils.mergMatrix(mat(hi_w),mat(hi_b)))
    
    # 输出层参数
    out_w = 2.0*(random.rand(nOut,nHidden)-0.5) 
    out_b = 2.0*(random.rand(nOut,1)-0.5)
    out_wb = mat(Untils.mergMatrix(mat(out_w),mat(out_b)))
    # 默认旧权值
    dout_wbOld = 0.0 ; dhi_wbOld = 0.0 

    for i in xrange(maxiter):   
        #1. 工作信号正向传播
        
        #1.1 输入层到隐含层
        hi_input = hi_wb*SampIn
        hi_output = logistic(hi_input)        
        hi2out  = Untils.mergMatrix(hi_output.T, ones((nSampNum,1))).T
        
    		#1.2 隐含层到输出层    		
        out_input = out_wb*hi2out
        out_output = logistic(out_input)
        
        #2. 误差计算     
        err = expected - out_output 
        sse = errorfunc(err)
        errlist.append(sse);
        #2.1 判断是否收敛
        if sse <= eb:
            print "iteration:",i+ 1    
            break;
        
        #3.误差信号反向传播
        #3.1 DELTA为输出层到隐含层梯度  
        DELTA = multiply(err,dlogit(out_input,out_output))
        wDelta = out_wb[:,:-1].T*DELTA 
        
        #3.2 delta为隐含层到输入层梯度
        delta = multiply(wDelta,dlogit(hi_input,hi_output))        
        dout_wb = DELTA*hi2out.T
        
        #3.3 输入层的权值更新
        dhi_wb = delta*SampIn.T    
        
        #3.4 更新输出层和隐含层权值
        if i == 0:  
            out_wb = out_wb + eta * dout_wb 
            hi_wb = hi_wb + eta * dhi_wb
        else :    
            out_wb = out_wb + (1.0 - mc)*eta*dout_wb  + mc * dout_wbOld
            hi_wb = hi_wb + (1.0 - mc)*eta*dhi_wb + mc * dhi_wbOld
        dout_wbOld = dout_wb
        dhi_wbOld = dhi_wb     
    return errlist,out_wb,hi_wb

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
         hi_input = wex*(mat(xi).T)
         hi_out = logistic(hi_input) 
         taumrow,taucol= shape(hi_out)
         tauex = mat(ones((1,taumrow+1)))
         tauex[:,0:taumrow] = (hi_out.T)[:,0:taumrow]
         HM = WEX*(mat(tauex).T)
         out = logistic(HM) 
         z[i,j] = out 
    return x,z
             