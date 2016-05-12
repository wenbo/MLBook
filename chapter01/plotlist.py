# -*- coding: utf-8 -*-

import sys  
import os
import numpy as np
import matplotlib.pyplot as plt

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# 曲线数据加入噪声

x = np.linspace(-5,5,200);
y = np.sin(x);# 给出y与x的基本关系
yn = y+np.random.rand(1,len(y))*1.5 ;	# 加入噪声的点集
# 绘图
fig = plt.figure()
ax = fig.add_subplot(111) 
ax.scatter(x,yn,c='blue',marker='o')
ax.plot(x,y+0.75,'r') 
plt.show()

