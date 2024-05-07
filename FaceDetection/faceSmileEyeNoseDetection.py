'''
Haar Cascade Face, Smile and Eye detection with OpenCV

Developed by Marcelo Rovai - MJRoBot.org @ 22Feb2018
'''

import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# faceCascade = cv2.CascadeClassifier를 사용해 'haarcascade_frontalface_default.xml'이라는 XML 파일을 사용하여
# 정면 얼굴을 감지하는 카스케이드 분류기를 초기화 한다.
# 카스케이드 분류기는 이미지나 비디오 프레임에서 얼굴을 감지하는 데 사용된다.
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# 'haarcascade_eye.xml'이라는 XML 파일을 사용하여 눈을 감지하는 카스케이드 분류기를 초기화한다.
# 이러한 카스케이드 분류기는 눈의 특징을 학습한 모델로, 이미지나 비디오 프레임에서 눈을 식별하는 데 사용된다.
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

# 'haarcascade_smile.xml'이라는 XML 파일을 사용하여 웃음을 감지하는 카스케이드 분류기를 초기화한다.
# 이러한 카스케이드 분류기는 웃음의 특징을 학습한 모델로, 이미지나 비디오 프레임에서 웃음을 식별하는 데 사용된다.
smileCascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')

noseCascade = cv2.CascadeClassifier('Cascades/haarcascade_mcs_nose 복사본.xml')

# 인덱스 0에 해당하는 비디오 캡처 장치를 연다는 의미이다. 일반적으로 이것은 컴퓨터에 연결된 첫 번째 웹캠을 가르킨다.
cap = cv2.VideoCapture(0)

# cap.set(3,640) 및 cap.set(4,480) 호출은 OpenCV의 비디오 캡처 객체에서 프레임의 너비와 높이를 각각 640 픽셀과 480 픽셀로 설정하는 것이다.

# cap.set(3,640): 이 호출은 비디오 캡처 객체의 속성 중 너비를 설정한다. 3은 너비를 나타내는 식별자이다.
# 여기서는 비디오 프레임의 너비를 640 픽셀로 설정하고 있다.

# cap.set(4,480): 이 호출은 비디오 캡처 객체의 속성 중 높이를 설정한다. 4는 높이를 나타내는 식별자이다.
# 여기서는 비디오 프레임의 높이를 480 픽셀로 설정하고 있다.
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

# while True문으로 내가 ESC를 눌러 종료하기 전까지 실행시킨다.
while True:

    #cap.read() 함수는 비디오 캡처 객체에서 다음 프레임을 읽어오는 작업을 수행한다.
    ret, img = cap.read()

    #cv2.flip() 함수는 이미지를 수평 또는 수직으로 뒤집는 데 사용된다. 이 함수는 다음 매개변수를 사용한다.
    img = cv2.flip(img, +1)

    #cv2.cvtColor() 함수는 이미지의 컬러 스페이스를 변환하는 데 사용된다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #faceCascade.detectMultiScale() 함수는 입력 이미지에서 객체(여기서는 얼굴)를 감지하는 데 사용된다.
    faces = faceCascade.detectMultiScale(
        #첫 번째 매개변수는 감지할 이미지이다.
        gray,
        #scaleFactor: 이미지 크기가 감소되는 스케일 팩터이다.
        scaleFactor=1.3,
        #minNeighbors: 각 후보 사각형은 이웃한 후보 사각형들에 의해 검증된다. 이웃 사각형은 이 값에 따라 얼마나 많이 반드시 검증되어야 하는지를 결정된다.
        minNeighbors=5,
        #minSize: 객체로 간주되기 위한 최소 사각형 크기이다. 이 값보다 작은 객체는 무시된다.
        minSize=(30, 30)
    )

    #faces 변수에 저장된 얼굴 영역의 각각에 대해 반복한다. faces 변수는 이전 단계에서 얼굴을 감지한 후 반환된 얼굴 사각형들의 리스트이다.
    for (x, y, w, h) in faces:
        #감지된 얼굴 주변에 파란색 사각형을 그립니다
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #얼굴 영역의 그레이스케일 이미지를 추출한다. gray 이미지는 이전 단계에서 변환된 그레이스케일 이미지이다.
        roi_gray = gray[y:y + h, x:x + w]
        #얼굴 영역의 컬러 이미지를 추출한다. img 이미지는 원본 컬러 이미지이다.
        roi_color = img[y:y + h, x:x + w]


        eyes = eyeCascade.detectMultiScale(
            #첫 번째 매개변수는 눈을 감지할 이미지이다. roi_gray 변수에 저장된 얼굴 영역의 그레이스케일 이미지가 사용된다.
            roi_gray,
            #이미지 크기가 감소되는 스케일 팩터이다.
            scaleFactor=1.5,
            #각 후보 사각형은 이웃한 후보 사각형들에 의해 검증된다.
            minNeighbors=5,
            #객체로 간주되기 위한 최소 사각형 크기이다.
            minSize=(5, 5),
        )

        #변수에 저장된 각각의 눈 영역에 대해 반복한다.
        #각 눈 영역은 (ex, ey, ew, eh) 형식으로 표현되며, 이는 각각 눈의 왼쪽 위 모서리의 좌표와 너비, 높이를 의미한다.
        for (ex, ey, ew, eh) in eyes:
            #각 눈 주변에 초록색 사각형을 그리고 이 사각형은 눈 영역을 시각적으로 표시하기 위한 것.
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        #eyes와 같은 형식이다.
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )

        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

        # eyes와 같은 형식이다.
        nose = noseCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(15, 15)
        )

        for (zx, zy, zw, zh) in nose:
            cv2.rectangle(roi_color, (zx, zy), (zx + zw, zy + zh), (0, 255, 0), 2)
#초록색 검출을 각각 다른 함수로 사용한 이유는 각 분류기는 서로 다른 객체를 감지하므로, 각 객체를 시각적으로 구분하기 위해
#각각 다른 반복문을 사용한다. 이렇게 하면 이미지에서 각 객체의 위치를 쉽게 식별하고 구분할 수 있기 때문이다.
#또한 서로 다른 객체를 나타내는 사각형을 그리는 것은 코드의 가독성을 높이고 유지 관리를 용이하게 만든다.

        #이미지를 화면에 표시하는 데 사용된다. 윈도우의 이름은 화면에 표시되는 이미지 창의 제목이 된다.
        #두 번째 매개변수는 표시할 이미지이다.
        cv2.imshow('video', img)

#cv2.waitKey(30) 함수는 30밀리초 동안 키 이벤트를 기다린다. 이 함수는 키 이벤트가 발생하면 해당 키의 ASCII 코드 값을 반환하고, 아무 키도 눌리지 않으면 -1을 반환한다.
#k = cv2.waitKey(30) & 0xff는 반환된 키 이벤트의 ASCII 코드 값을 k 변수에 저장한다. 이때 0xff와의 비트 AND 연산을 수행하여 마지막 8비트(1바이트)만 사용한다.
# 이는 키 이벤트 값이 0~255 사이의 값을 갖도록 보장한다.
    k = cv2.waitKey(30) & 0xff
#if k == 27:는 ESC(ASCII 코드 값이 27) 키가 눌렸는지 확인한다. ESC 키가 눌리면 반복문을 종료하고 프로그램이 종료된다.
    if k == 27:  # press 'ESC' to quit
        break
#따라서 ESC키가 눌릴 때까지 프로그램이 계속해서 비디오를 표시하고, ESC키가 눌리면 프로그램이 종료되도록 한다.
cap.release()
cv2.destroyAllWindows()
