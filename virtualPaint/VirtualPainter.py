import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################
brushThickness = 25
eraserThickness = 100
########################


folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)
shape = 'freestyle'

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# frame width and hight setting
cap.set(3, 1280)  # 806
cap.set(4, 720)  # (cv2.CAP_PROP_FRAME_HEIGHT, 360)
# print(cap.get(4))

detector = htm.handDetector(detectionCon=0.85)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    # img = cv2.resize(img, (806, 720))
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbs = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList[4]) # lmList[4]
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        xt, yt = lmList[4][1:]
        dist = int((((yt - y1) ** 2) + ((xt - x1) ** 2)) ** 0.5)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            # xp, yp = 0, 0
            # print("Selection Mode")
            # # Checking for the click
            if y1 < 125:
                if 200 < x1 < 400:
                    header = overlayList[0]
                    drawColor = (64, 64, 128)
                elif 450 < x1 < 650:
                    header = overlayList[1]
                    drawColor = (0, 128, 255)
                elif 700 < x1 < 900:
                    header = overlayList[2]
                    drawColor = (64, 128, 0)
                elif 1000 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            # cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        cv2.circle(img, (x1, y1), 25, drawColor, -1)
        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            # cv2.circle(img, (x1, y1), 15, drawColor, -1)
            # print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            #
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

        xp, yp = x1, y1

        # # Clear Canvas when all fingers are up
        if all(x >= 1 for x in fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    # img = cv2.resize(img, (806, 720))
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    # cv2.waitKey(1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

