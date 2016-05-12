# -*- coding: utf-8 -*-

import sys  
import os
import time
from numpy import *

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

# 按行读文件，读取指定行数
# nmax=0按行读取全部
def readfilelines(path,nmax=0):
	fp = open(path,"rb")
	ncount = 0 # 已读取行
	while True:
		content = fp.readline()			
		# 判断是否到文件尾，是否读取到
		if content =="" or (ncount>=nmax and nmax!=0):
			break
		yield content  # 返回读取的行
		if nmax != 0 : 	ncount += 1
	fp.close()
	
path = "testdata/01.txt" #数据文件所在路径
# 读取10行,
for line in readfilelines(path,nmax=10):
	print line.strip()


