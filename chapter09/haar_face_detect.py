# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('E:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml')

img = cv2.imread('mypicture.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 识别输入图片中的人脸对象.返回对象的矩形尺寸
# 函数原型detectMultiScale(gray, 1.2,3,CV_HAAR_SCALE_IMAGE,Size(30, 30))
# gray需要识别的图片
# 1.2：表示每次图像尺寸减小的比例
# 3：表示每一个目标至少要被检测到4次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)
# CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(30, 30)为目标的最小最大尺寸
# faces：表示检测到的人脸目标序列
faces = face_cascade.detectMultiScale(gray, 1.2, 3)
for (x,y,w,h) in faces:
	img2 = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),4)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("paulwalker.head.jpg", img) # 保存图片