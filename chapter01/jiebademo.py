#encoding=utf-8
import sys
import jieba


seg_list = jieba.cut("来到北京清华大学", cut_all=False)
print "Default Mode:", "/ ".join(seg_list)  # 默认模式

seg_list = jieba.cut("他来到网易杭研大厦")
print " ".join(seg_list)

