# -*- coding: GBK -*-
# Filename :gradDecent.py

from numpy import *
import operator
import Untils
import BackPropgation
import matplotlib.pyplot as plt 

# BP神经网络: XOR实例

# 数据集
dataSet = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
classLabels = [0,1,1,0]

# 数据集矩阵化
SampIn = mat(dataSet).transpose()
expected = mat(classLabels)
# 网络参数
eb = 0.01                   # 误差容限 
eta = 0.6                   # 学习率 
mc = 0.8                    # 动量因子 
maxiter = 1000              # 最大迭代次数 
itera = 0                   # 第一代

# 构造网络

# 初始化网络
nSampNum = 4;  # 样本数量
nSampDim = 2;  # 样本维度
nHidden = 3;   # 隐含层神经元 
nOut = 1;      # 输出层

# 输入层参数

# 隐含层参数
# net_Hidden * 3 一行代表一个隐含层节点
w = 2*(random.rand(nHidden,nSampDim)-1/2)  
b = 2*(random.rand(nHidden,1)-1/2) 
wex = mat(Untils.mergMatrix(mat(w),mat(b)))

# 输出层参数
W = 2*(random.rand(nOut,nHidden)-1/2) 
B = 2*(random.rand(nOut,1)-1/2) 
WEX = mat(Untils.mergMatrix(mat(W),mat(B)))

dWEXOld = 0 ; dwexOld = 0 
# 训练
iteration = 0;  
errRec = [];
for i in range(maxiter):   
    # 工作信号正向传播
    hp = wex*SampIn
    tau = BackPropgation.logsig(hp)
    tauex  = Untils.mergMatrix(tau.T, ones((nSampNum,1))).T

    HM = WEX*tauex
    out = BackPropgation.logsig(HM)    
    err = expected - out 
    sse = BackPropgation.sumsqr(err) 
    errRec.append(sse); 
    # 判断是否收敛
    iteration = iteration + 1    
    if sse <= eb:
        print "iteration:",i    
        break;
     
    # 误差信号反向传播
    # DELTA和delta为局部梯度  
    DELTA = multiply(err,BackPropgation.dlogsig(HM,out))
    wDelta = W.T*DELTA
    delta = multiply(wDelta,BackPropgation.dlogsig(hp,tau))
    dWEX = DELTA*tauex.T
    dwex = delta*SampIn.T       
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

# 重构dataSet数据集
dataMat = mat(ones((shape(dataSet)[0],shape(dataSet)[1])))
dataMat[:,1] = mat(dataSet)[:,0]
dataMat[:,2] = mat(dataSet)[:,1]	

# 绘制数据点
Untils.drawClassScatter(dataMat,transpose(expected))

# 绘制分类线


# 绘制误差曲线
X = linspace(0,1000,1000)
Untils.TrendLine(X,errRec)