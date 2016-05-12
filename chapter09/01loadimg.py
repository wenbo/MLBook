# -*- coding: utf-8 -*-

from numpy import *
import cv2

win_name = 'mypicture' #窗口名称
# cv2.WINDOW_NORMAL:可以手动调整窗口大小
cv2.namedWindow( win_name , cv2.WINDOW_NORMAL)
img = cv2.imread( 'snapshot0001.jpg',0) #0 黑白图片；1 原色图片
cv2.imshow(win_name ,img) # 显示图片
cv2.waitKey(0)
cv2.destroyAllWindows() # 销毁创建的对象
# 保存图片
# cv2.imwrite("paulwalker.mono.pgm", img)