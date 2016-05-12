# -*- coding: utf-8 -*-

import sys  
import os
from numpy import * 
import matplotlib.pyplot as plt
# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

base = mat([[3,1],[1,3]])
v1 = mat([1,2])
print linalg.norm(v1)
print (base[0]*base[1].T)/(linalg.norm(base[1])*linalg.norm(base[0]))
v2 = v1*base
print v2
print linalg.norm(v2)

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111)
x0 = linspace(0,1,200)
y0 = 2*x0
x1 = linspace(0,1,200)
y1 = 3*x1
x2 = linspace(0,3,200)
y2 = x2/3 
x3 = linspace(0,5,200)
y3 = 7*x3/5
ax.plot(x0,y0,"r")
plt.annotate("(1,2)",xy = (1,2))	
ax.plot(x1,y1,"b")
plt.annotate("(1,3)",xy = (1,3))	
ax.plot(x2,y2,"b")
plt.annotate("(3,1)",xy = (3,1))	
ax.plot(x3,y3,"r")
plt.annotate("(5,7)",xy = (5,7))	
#平行四边形
x7 = linspace(0,1,200)
y7 = linspace(2,2,200)
ax.plot(x7,y7,"b",linestyle='--')	
x8 = linspace(1,1,200)
y8 = linspace(0,2,200)
ax.plot(x8,y8,"b",linestyle='--')	
x4 = linspace(1,2,200)
y4 = 3*x4
ax.plot(x4,y4,"b",linestyle='--')	
x5 = linspace(2,5,200)
y5 = 6+x0
ax.plot(x5,y5,"b",linestyle='--')
x6 = linspace(3,5,200)
y6 = 1+6*x0
ax.plot(x6,y6,"b",linestyle='--')
plt.xlim(0, 8)
plt.ylim(0, 8)
plt.grid(True)
plt.show()


