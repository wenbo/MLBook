# -*- coding: utf-8 -*-

from numpy import *
import cv2


# Create a black image
img=zeros((512,512,3))
# Draw a diagonal blue line with thickness of 5 px
# 起点:(0,0),终点:(511,511)，颜色:( 255,0,0)，宽度:2
cv2.line(img,(0,0),(511,511),( 255,0,0),2)
cv2.imshow('image' ,img)
cv2.waitKey(0)
cv2.destroyAllWindows()