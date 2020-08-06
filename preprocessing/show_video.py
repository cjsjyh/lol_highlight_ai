import cv2
import numpy as np

capture = cv2.VideoCapture("/home/lol/tj/20200628_AF_DRX_1_highlight.mp4")

while True:
    frame = input("insert frame number to see : ")
    capture.set(cv2.CAP_PROP_POS_FRAMES,int(frame))
    ret, image = capture.read()
    cv2.imshow("image",image)
    res = cv2.waitKey(5)
    if res == 'q':
        break


