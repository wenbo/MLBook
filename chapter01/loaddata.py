# -*- coding: utf-8 -*-

import sys  
import os
import time
from numpy import *

# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')

#数据文件转矩阵
# path: 数据文件路径
# delimiter: 文件分隔符
def file2matrix(path,delimiter):	
	recordlist = []
	fp = open(path,"rb") 	# 读取文件内容
	content = fp.read()
	fp.close()
	rowlist = content.splitlines() 	# 按行转换为一维表
	# 逐行遍历
	# 结果按分隔符分割为行向量	
	recordlist =[ row.split(delimiter) for row in rowlist if row.strip()]
	return mat(recordlist)	# 返回转换后的矩阵形式
	
root = "testdata" #数据文件所在路径
pathlist = os.listdir(root) # 获取路径下所有数据文件
for path in pathlist:
	recordmat = file2matrix(root+"/"+path,"\t") # 文件到矩阵的转换
	print shape(recordmat) # 输出解析后矩阵的行、列数


