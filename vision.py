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


CALIBRATE = False
stopcalib = True
calibdatastruct = []
datafile = open("calibrationdata.pkl", "wb")
calibrationtargetdist = 36 #inches
distbetweencams = 12 #inches

#UDP stuff
VISION_TARGET = "192.168.0.3"
VISION_PORT = 1311

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# vision information (sans struct)
version = 1
valid = 0
ljy = 0
rjx = 0
rjy = 0
rljy = 0           # we will use ints only for vision, send zeros for floats
rrjx = 0
rrjy = 0

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


# Camera 2 Setup
def initCapture2():
    print("Initializing Video Capture Two...")
    cap2 = cv2.VideoCapture(1)
    if not cap2.isOpened():
        print("    failed to open camera device 1")
        print("    resetting device id to 1")
        initCapture2()
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    global exposure
    #exposure = cap2.get(cv2.CAP_PROP_EXPOSURE)
    cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure & white balance
    cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    print("    done.")
    return cap2


# HSV filters for green (input is an image)
def greenprocess(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, green_lower, green_upper)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    return contours, img


# HSV filters for orange ball (input is an image)
def orangeprocess(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, orange_lower, orange_upper)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10)))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    return contours, img


# does it divide by zero
def divbyzero(coord1, coord2):
    if (coord1 - coord2) == 0 or (coord2 - coord1) == 0:
        return True
    else:
        return False


# finds center of 2 green contours
def greencenter(contours):
    areaArray = []
    count = 1

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    largestcontour = sorteddata[0][1]
    secondlargestcontour = sorteddata[1][1]

    _ , y1 , _ , h1 = cv2.boundingRect(largestcontour)
    _ , y2 , _ , h2 = cv2.boundingRect(secondlargestcontour)

    y1 = y1 + (h1 / 2)
    y2 = y2 + (h2 / 2)
    yc = (y1 + y2) / 2
    """
    f = 1
    while xc < close and xc > far:
        secondlargestcontour = sorteddata[f+1][1]
        try:
            x2, y2, w2, h2 = cv2.boundingRect(secondlargestcontour)
            xc = (x1 + x2)/2
        except:
            pass

    return xc, yc 
    """
    return yc

def savefsandclose(fsnew):
    file = open("config.json", "w")

    jsonObj = {
            'values': {
                'savedfs': str(fsnew)
            },
            'camera': {
                'camdistance': str(distbetweencams)
            }
        }

    file.write(u"" + json.dumps(jsonObj))
    file.close()

def loadfsromFile():
    try:
        with open("config.json") as json_data:
            data = json.load(json_data)
            return np.array([data["values"]["savedfs"], data["camera"]["camdistance"]], dtype=np.uint8)
    except IOError as e:
        return np.array([2.5, 36], dtype=np.uint8)


def xdiff(xuno, xdos):
    if xuno != None and xdos != None:
        if xuno - xdos > 0:
            diff = (xuno - xdos)
            return diff

        if xdos - xuno > 0:
            diff = (xdos - xuno)
            return diff
    else:
        diff = None
        return diff

def zcalc(diff, fs):
    if diff == 0 or diff == None:
        z = None
    else:
        z = (distbetweencams * fs) / diff
    return z


# FIXED #fix to get angle too...
def distancenangle(z, offset):
    if z == None:
        angle = None
    else:
        angle = math.degrees(math.atan(offset / z))
    return z, angle

def calcoffset(x2, z,):
    if x2 != None and z != None:
        offset = ((x2*z)/savedfs)-(distbetweencams/2)
        return offset
    else:
        offset = None
        return offset


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
orange_upper = np.array([115,139,255], np.uint8)
orange_lower = np.array([90,84,128], np.uint8)

global cap1
global cap2

cap1 = initCapture1()
cap2 = initCapture2()

############################## Calibration Loop ####################################
while (stopcalib == False):

    # FIXED (SORTA) # this is a really bad way to do this, someone help fix pls

    # This section calibrates if signal from rio
    # Checks for UDP signal from RIO that says to calibrate

    listen()

    if (CALIBRATE == True):
        stopcalib = False
    else:
        stopcalib = True

    ret, img1 = cap1.read()
    ret, img2 = cap2.read()

    green1conts, output1green = greenprocess(img1)
    green2conts, output2green = greenprocess(img2)

    # Finding Contours
    if len(green1conts) > 0 and len(green2conts) > 0:
        cv2.drawContours(output1green, green1conts, -1, 255, 3)
        cont1max = max(green1conts, key=cv2.contourArea)
        xc1, yc1, wc1, hc1 = cv2.boundingRect(cont1max)
        camcoord1 = yc1

        cv2.drawContours(output2green, green2conts, -1, 255, 3)
        cont2max = max(green2conts, key=cv2.contourArea)
        xc2, yc2, wc2, hc2 = cv2.boundingRect(cont2max)
        camcoord2 = yc2

        # Stereo Calibration
        if (divbyzero(camcoord2, camcoord1) == False):
            fsnew = (calibrationtargetdist*(camcoord1-camcoord2))/distbetweencams
            print("fs is... " + str(fsnew))
            savefsandclose(fsnew)
            """
            with open('offsetdata.txt') as a:
                newTextoffset=a.read().replace(lastoffset, str(offsetnew)) # Replaces the previous calibration data
            with open('offsetdata.txt', "w") as a:
                    a.write(newTextoffset)

            with open('fsdata.txt') as b:
                    newTextfs=b.read().replace(lastfs, str(fsnew))
            with open('fsdata.txt', "w") as b:
                    b.write(newTextfs)
            """

            print("Calibration Complete")
            stopcalib = True

        ############################## Detection Loop ####################################

# grabbing the pickles...



#datafile = open("calibrationdata.pkl", "rb")
#calibdata = pickle.load(datafile)
#datafile.close()

calibdatastruct = loadfsromFile()
savedfs = calibdatastruct[0]

print("last focal value was " + str(savedfs))
try:
    while True:
        ret, img1 = cap1.read()
        ret, img2 = cap2.read()

        green1conts, output1green = greenprocess(img1)
        green2conts, output2green = greenprocess(img2)
        orange1conts, output1orange = orangeprocess(img1)
        orange2conts, output2orange = orangeprocess(img2)

        # are there more than 2 tape targets?
        if len(green1conts) >= 2 and len(green2conts) >= 2:
            greenx1, greeny1 = greencenter(green1conts)
            greenx2, greeny2 = greencenter(green2conts)
        else:
            greenx1 = None
            greeny1 = None
            greenx2 = None
            greeny2 = None

        # is there one ball?
        if len(orange1conts) > 0:
            o1 = max(orange1conts, key=cv2.contourArea)
            orangex1, orangey1, orangew1, orangeh1 = cv2.boundingRect(o1)
        else:
            orangex1 = None
            orangey1 = None
            orangew1 = None
            orangeh1 = None

        if len(orange2conts) > 0:
            o2 = max(orange2conts, key=cv2.contourArea)
            orangex2, orangey2, orangew2, orangeh2 = cv2.boundingRect(o2)
        else:
            orangex2 = None
            orangey2 = None
            orangew2 = None
            orangeh2 = None

        ############################## Stereo Calculations #################################

        orangediff = xdiff(orangex1, orangex2)
        greendiff = xdiff(greenx1, greenx2)

        orangez = zcalc(orangediff, savedfs)
        greenz = zcalc(greendiff, savedfs)

        orangeoffset = calcoffset(orangex2, orangez)
        greenoffset = calcoffset(greenx2, greenz)

        orangedist, orangeangle = distancenangle(orangez, orangeoffset)
        greendist, greenangle = distancenangle(greenz, greenoffset)

        print(exposure)
        ############################## UDP Stuff ###########################################

        send()
        cv2.imshow('orange1', output1orange)
        cv2.imshow('orange2', output2orange)

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
