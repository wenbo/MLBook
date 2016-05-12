# -*- coding: utf-8 -*-

from numpy import *
import cv2
img=zeros(( 512, 512, 3))
cv2.rectangle(img,( 384, 0),( 510, 128),( 0, 255, 0), 3) #矩形
cv2.circle(img,( 447, 63), 63, ( 0, 0, 255), - 1) #圆
cv2.ellipse(img,( 256, 256),( 100, 50), 0, 0, 360, 255, -1) #椭圆
#画多边形
pts=array([[10,5],[ 20,30],[70,20],[50,10]])
cv2.polylines(img,[pts],True,(0,255,255),1)
#写入文字
font=cv2.FONT_HERSHEY_SIMPLEX
cv2. putText(img, 'OpenCV' ,( 10, 500), font, 4,( 255, 255, 255), 2)
cv2.imshow('image' ,img)
cv2.waitKey(0)
cv2.destroyAllWindows()