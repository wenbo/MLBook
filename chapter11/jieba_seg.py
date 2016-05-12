# -*- coding: utf-8 -*-
#encoding=utf-8

import sys  
import jieba
import jieba.posseg as pseg 
reload(sys)
sys.setdefaultencoding('utf-8')

seg_list = pseg.cut("把这篇文章修改一下。")  # 默认是精确模式

for word in seg_list:
	print word.word," ",word.flag


