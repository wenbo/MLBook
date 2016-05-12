# -*- coding: utf-8 -*-

import sys  
import os
from numpy import *
# 设置utf-8 unicode环境
reload(sys)
sys.setdefaultencoding('utf-8')

labels=["年龄","收入","学生","信誉"]
dataset = [[0,0,0,0,"no"],[0,0,0,1,"no"],[0,1,0,0,"no"],[0,2,1,0,"yes"],[0,1,1,1,"yes"],
[1,0,0,0,"yes"],[1,2,1,1,"yes"], [1,1,0,1,"yes"],[1,0,1,0,"yes"],[2,1,0,0,"yes"],[2,2,1,0,"yes"],
[2,2,1,1,"no"],[2,1,1,0,"yes"],[2,1,0,1,"no"]]
numlist = [64	,64	,128,64	,64	,128,64	,32	,32	,60	,64	,64	,132,64	 ]
print mat(dataset).T
datalines =[]

for element,num in zip(dataset,numlist):
	liststr =""
	for cell in element:
		liststr += str(cell)+"\t"
	liststr = liststr[:-1]	
	for i in xrange(num):
		datalines.append(liststr)
	
fp = open("dataset.dat","w")
fp.write("\n".join(datalines)) 
                    