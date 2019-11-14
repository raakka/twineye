#imports and stuff
import cv2
import numpy as np
import math import time

#def constants
stop_calibration = True


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


#calibration for stero values...
#during calibration target should be 1 meter away from the center 

while stop_calibration == False:

    ret, img1 cap1.read()
    ret, img2 cap2.read()

    blur1