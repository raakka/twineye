from datetime import datetime
import cv2
import numpy as np
import socket
import time
import math
import json
import binascii
import struct
import sys

############# Defining Variables and Tuning #############

#UDP stuff
VISION_TARGET = "192.168.0.2"
VISION_PORT = 1311

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# vision information (sans struct)
version = 1
valid = 1
angle = 0
distance = 0
placeholder1 = 0
placeholder2 = 0           # we will use ints only for vision ( -32767 to 32767 ), send zeros for floats
placeholder3 = 0
placeholder4 = 0

# # # # # # # # # # # # # # DEEP SPACE # # # # # # # # # #
#                 ____
#                /___.`--.____ .--. ____.--(
#                       .'_.- (    ) -._'.
#                     .'.'    |'..'|    '.'.
#              .-.  .' /'--.__|____|__.--'\ '.  .-.
#             (O).)-| |  \    |    |    /  | |-(.(O)
#              `-'  '-'-._'-./      \.-'_.-'-'  `-'
#                 _ | |   '-.________.-'   | | _
#              .' _ | |     |   __   |     | | _ '.
#             / .' ''.|     | /    \ |     |.'' '. \
#             | |( )| '.    ||      ||    .' |( )| |
#             \ '._.'   '.  | \    / |  .'   '._.' /
#              '.__ ______'.|__'--'__|.'______ __.'
#             .'_.-|         |------|         |-._'.
#            //\\  |         |--::--|         |  //\\
#           //  \\ |         |--::--|         | //  \\
#          //    \\|        /|--::--|\        |//    \\
#         / '._.-'/|_______/ |--::--| \_______|\`-._.' \
#        / __..--'        /__|--::--|__\        `--..__ \
#       / /               '-.|--::--|.-'               \ \
#      / /                   |--::--|                   \ \
#     / /                    |--::--|                    \ \
# _.-'  `-._                 _..||.._                  _.-` '-._
# '--..__..--'               '-.____.-'                '--..__..-'

############ Functions for Processing ############

# Camera 1 Setup
def initCapture1():
    print("Initializing Video Capture One...")
    cap1 = cv2.VideoCapture(0)
    if not cap1.isOpened():
        print("    failed to open camera device 0")
        print("    resetting device id to 0")
        initCapture1()
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    global exposure
    exposure = cap1.get(cv2.CAP_PROP_EXPOSURE)
    cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure & white balance
    cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 3) # Small buffersize, helps latency
    print("    done.")
    return cap1


# HSV filters img input
def getcontours(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, green_lower, green_upper)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10)))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    return contours, img

# finds center of 2 green contours
def greencenter(contours):
    areaArray = []
    count = 1
        
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    largestcontour = sorteddata[0][1]
    m = cv2.moments(largestcontour)
    cx = int(M['m10']/M['m00'])
    return cx

# FIXED #fix to get angle too...
def angle(cx):
    if z == None:
        angle = None
    else:
        angle = math.degrees(math.atan(cx - 320) / 320))
    return angle

# listens for signal from RIO that says to calibrate (change CALIBRATE = True)
def listen():
    print("this does nothing rn")

def send():
    try:
        values = (version, valid, ljy, rjx, rjy, rljy, rrjx, rrjy)
        packer = struct.Struct('!i i i i i f f f')  # the ! implements network byte order for the payload
        packed_data = packer.pack(*values)
        retval = sock.sendto(packed_data, (VISION_TARGET, VISION_PORT))
        #time.sleep(1)
    finally:
        pass

# Not functions...
# USE THE TUNER TO ADJUST FOR THESE!!!!
green_upper = np.array([85, 200, 200], np.uint8)
green_lower = np.array([75, 120, 120], np.uint8)

global cap1

cap1 = initCapture1()

############################## Detection Loop ####################################

try:
    while True:
        ret, img1 = cap1.read()
        conts, output = process(img1)
        
############################## math voodoo #################################
        cx = greencenter(conts)
        angle = angle(cx)
        print(exposure)
        
        try:
		    values = (version, valid, angle, distance, placeholder1, placeholder2, placeholder3, placeholder4)
		    packer = struct.Struct('!f f f f f f f f')   # the ! implements network byte order for the payload
		    packed_data = packer.pack(*values)
		    retval = sock.sendto(packed_data, (VISION_TARGET, VISION_PORT))
		    #time.sleep(1)
            
############################## UDP Stuff ###########################################
        cv2.imshow('output', output)
        
        k = cv2.waitKey(10) & 255
        if k == 27:
            break
        # W key: Increment exposure.
        elif k == ord('w'):
            exposure += 1
        # S key: Decrement exposure.
        elif k == ord('s'):
            exposure -= 1

        if exposure < -7:
            exposure = -7
        elif exposure > -1:
            exposure = -1

finally:
    sock.close()

cap1.release()
cap2.release()
cv2.destroyAllWindows
