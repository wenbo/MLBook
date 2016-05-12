# -*- coding: utf-8 -*-

from numpy import *
import cv2
img=zeros(( 512, 512, 3))
# 绘制圆：圆心(255, 255), 半径60, 颜色( 0, 255, 255), 像素1
cv2.circle(img,(255, 150), 60, ( 0, 255, 255), 2) #圆
# 绘制椭圆
# 中心点的位置(255, 255), 短半径50,长半径100  
# 360表示整个椭圆；颜色 0, 255, 255；像素2；
cv2.ellipse(img,(255, 350),(100, 50), 0, 0, 360, (255,255,0), 2) #椭圆
cv2.imshow('image' ,img)
cv2.waitKey(0)
cv2.destroyAllWindows()
