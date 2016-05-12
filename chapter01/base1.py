# -*- coding: utf-8 -*-

import sys  
import os
from numpy import * 
import matplotlib.pyplot as plt
# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

base = mat([[1,3],[3,1]])
print base[0]+base[1]

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111)
x1 = linspace(0,1,200)
y1 = 3*x1
x2 = linspace(0,3,200)
y2 = x2/3 
x3 = linspace(0,4,200)
y3 = x3
ax.plot(x1,y1,"b")
plt.annotate("(1,3)",xy = (1,3))	
ax.plot(x2,y2,"b")
plt.annotate("(3,1)",xy = (3,1))	
ax.plot(x3,y3,"b")
plt.annotate("(4,4)",xy = (4,4))	
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.grid(True)
plt.show()


