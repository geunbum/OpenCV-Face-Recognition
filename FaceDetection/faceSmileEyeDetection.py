'''
Haar Cascade Face, Smile and Eye detection with OpenCV  

Developed by Marcelo Rovai - MJRoBot.org @ 22Feb2018 
'''

import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
#faceCascade = cv2.CascadeClassifier를 사용해 'haarcascade_frontalface_default.xml'이라는 XML 파일을 사용하여
# 정면 얼굴을 감지하는 카스케이드 분류기를 초기화 한다.
#카스케이드 분류기는 이미지나 비디오 프레임에서 얼굴을 감지하는 데 사용된다.
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

#'haarcascade_eye.xml'이라는 XML 파일을 사용하여 눈을 감지하는 카스케이드 분류기를 초기화한다.
#이러한 카스케이드 분류기는 눈의 특징을 학습한 모델로, 이미지나 비디오 프레임에서 눈을 식별하는 데 사용된다.
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

#'haarcascade_smile.xml'이라는 XML 파일을 사용하여 웃음을 감지하는 카스케이드 분류기를 초기화한다.
#이러한 카스케이드 분류기는 웃음의 특징을 학습한 모델로, 이미지나 비디오 프레임에서 웃음을 식별하는 데 사용된다.
smileCascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')

#인덱스 0에 해당하는 비디오 캡처 장치를 연다는 의미이다. 일반적으로 이것은 컴퓨터에 연결된 첫 번째 웹캠을 가르킨다.
cap = cv2.VideoCapture(0)

#cap.set(3,640) 및 cap.set(4,480) 호출은 OpenCV의 비디오 캡처 객체에서 프레임의 너비와 높이를 각각 640 픽셀과 480 픽셀로 설정하는 것이다.

#cap.set(3,640): 이 호출은 비디오 캡처 객체의 속성 중 너비를 설정한다. 3은 너비를 나타내는 식별자이다.
# 여기서는 비디오 프레임의 너비를 640 픽셀로 설정하고 있습니다.

#cap.set(4,480): 이 호출은 비디오 캡처 객체의 속성 중 높이를 설정한다. 4는 높이를 나타내는 식별자이다.
# 여기서는 비디오 프레임의 높이를 480 픽셀로 설정하고 있습니다.
cap.set(3,640) # set Width
cap.set(4,480) # set Height

#while True문으로 내가 ESC를 눌러 종료하기 전까지 실행시킨다.
while True:
    ret, img = cap.read()
    img = cv2.flip(img, +1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,      
        minSize=(30, 30)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            )
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
               
        
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )
        
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        
        cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
