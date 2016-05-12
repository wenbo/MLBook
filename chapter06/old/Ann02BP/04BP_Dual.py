# -*- coding: GBK -*-
# Filename :gradDecent.py

from numpy import *
import operator
import Untils
import BackPropgation
import matplotlib.pyplot as plt 

# BP神经网络

# 数据集: 列1:截距 1列2:x坐标 列3:y坐标
dataMat,classLabels = BackPropgation.loadDataSet() # 初始化时第1列为全1向量
[m,n] = shape(dataMat) 
SampIn = mat(BackPropgation.normalize(mat(dataMat)).transpose())
expected = mat(classLabels)

# 网络参数
eb = 0.01                   # 误差容限 
eta = 0.6                   # 学习率 
mc = 0.8                    # 动量因子 
maxiter = 1000              # 最大迭代次数 

# 构造网络

# 初始化网络
nSampNum = m;  # 样本数量
nSampDim = 2;  # 样本维度
nHidden = 3;   # 隐含层神经元 
nOut = 1;      # 输出层

# 隐含层参数
# net_Hidden * 3 一行代表一个隐含层节点
w = 2*(random.rand(nHidden,nSampDim)-1/2)  
b = 2*(random.rand(nHidden,1)-1/2) 
wex = mat(Untils.mergMatrix(mat(w),mat(b)))

# 输出层参数
W = 2*(random.rand(nOut,nHidden)-1/2) 
B = 2*(random.rand(nOut,1)-1/2) 
WEX = mat(Untils.mergMatrix(mat(W),mat(B)))

dWEXOld = [] ; dwexOld = [] # 初始化权值中间变量
# 训练
iteration = 0;  
# 初始化误差变量
errRec = [];

for i in range(maxiter):   
    # 工作信号正向传播
    hp = wex*SampIn
    tau = BackPropgation.logsig(hp)
    tauex  = Untils.mergMatrix(tau.transpose(), ones((nSampNum,1))).transpose()

    HM = WEX*tauex
    out = BackPropgation.logsig(HM)    
    
    err = expected - out 
    sse = BackPropgation.sumsqr(err) 
    errRec.append(sse)
    # 判断是否收敛
    iteration = iteration + 1
        
    if sse <= eb:
        print "iteration:",i 
        break
     
    # 误差信号反向传播
    # DELTA和delta为局部梯度  
    DELTA = multiply(err,BackPropgation.dlogsig(HM,out))
    wDelta = W.transpose()*DELTA
    delta = multiply(wDelta,BackPropgation.dlogsig(hp,tau))
    dWEX = DELTA*tauex.transpose()
    dwex = delta*SampIn.transpose()        
    # 更新权值
    if i == 0:  
        WEX = WEX + eta * dWEX
        wex = wex + eta * dwex
    else :    
        WEX = WEX + (1 - mc)*eta*dWEX + mc * dWEXOld
        wex = wex + (1 - mc)*eta*dwex + mc * dwexOld
 
    dWEXOld = dWEX
    dwexOld = dwex 
    W  = WEX[:,0:nHidden]

# 绘制误差曲线
X = linspace(0,1000,1000)
Untils.TrendLine(X,errRec)
   