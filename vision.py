#imports and stuff
import cv2
import numpy as np
import math import time

#def constants
stop_calibration = True

b5142
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VidepCapture(1)

cap1.set(4,480)
cap1.set(3,640)
cap1.set(15,0.1)
cap1.set(5,30)

cap2.set(4,480)
cap2.set(3,640)
cap2.set(15,0.1)
cap2.set(5,30)

orange_upper_colors = np.array([38,255,255])
orange_lower_colors = np.array([5,180,125])

green_upper_colors = np.array([h, s, v])
green_lower_colors = np.array([h, s, v])


#calibration for stero values...
#during calibration target should be 1 meter away from the center 

while stop_calibration == False:

    ret, img1 = cap1.read()
    ret, img2 = cap2.read()

    blur1 = cv2.GaussianBlur(img1, (11, 11), 0)
    blur2 = cv2.GaussianBlur(img2, (11, 11), 0)

    hsvconv1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2HSV)
    hsvconv2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)

    mask1_green = cv2.inRange(hsvconv1, green_lower_colors, green_upper_colors)
    mask2_green = cv2.inRange(hsvconv2, green_lower_colors, green_upper_colors)


