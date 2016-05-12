# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('E:\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml')

img = cv2.imread('snapshot0001.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.2, 3)
for (x,y,w,h) in faces:
	img2 = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),4)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("paulwalker.head.jpg", img) # 保存图片