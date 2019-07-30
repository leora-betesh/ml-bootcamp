# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:31:47 2019

@author: Leora Betesh

Taken from the tutorial at https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
"""

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
