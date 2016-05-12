# -*- coding: utf-8 -*-

import sys  
import os
import time

# ÅäÖÃutf-8Êä³ö»·¾³
reload(sys)
sys.setdefaultencoding('utf-8')

mylist = [1,2,3,4,5]
length = len(mylist)
a = 10
for indx in xrange(length):
	mylist[indx] = a*mylist[indx]
print mylist

