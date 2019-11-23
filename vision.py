from datetime import datetime
import cv2
import numpy as np
import socket
import time
import math
import array
import pickle

############# Defining Variables and Tuning #############

ip = "10.13.11.22"  # Change this number
port = 1311  # Change this number too
CALIBRATE = False
stopcalib = True
calibdatastruct = []
datafile = open("calibrationdata.pkl", "wb")
calibrationtargetdist = 36 #inches
distbetweencams = 36 #inches

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
    cap1.set(cv2.CAP_PROP_EXPOSURE, exposure)  # Disable auto exposure & white balance
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
    exposure = cap2.get(cv2.CAP_PROP_EXPOSURE)
    cap2.set(cv2.CAP_PROP_EXPOSURE, exposure)  # Disable auto exposure & white balance
    cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    print("    done.")
    return cap2


# HSV filters for green (input is an image)
def greenprocess(image):

    blur = cv2.GaussianBlur(image, (11, 11), 0)
    hsvconv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsvconv, green_lower, green_upper)
    mask_green = cv2.erode(mask_green, None, iterations=2)
    mask_green = cv2.dilate(mask_green, None, iterations=2)

    outputgreen = cv2.bitwise_and(image, image, mask=mask_green)
    ret, threshgreen = cv2.threshold(mask_green, 40, 255, 0)
    contgreen, _ = cv2.findContours(threshgreen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contgreen, outputgreen


# HSV filters for orange ball (input is an image)
def orangeprocess(image):

    blur = cv2.GaussianBlur(image, (11, 11), 0)
    hsvconv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask_orange = cv2.inRange(hsvconv, orange_lower, orange_upper)
    mask_orange = cv2.erode(mask_orange, None, iterations=2)
    mask_orange = cv2.dilate(mask_orange, None, iterations=2)

    outputorange = cv2.bitwise_and(image, image, mask=mask_orange)
    ret, threshorange = cv2.threshold(mask_orange, 40, 255, 0)
    contorange, _ = cv2.findContours(threshorange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contorange, outputorange


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

    return yc

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


def xdiff(xuno, xdos):
    if xuno != None or xdos != None:
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
    if x2 != None or z != None:
        offset = ((x2*z)/savedfs)-(distbetweencams/2)
        return offset
    else:
        offset = None
        return offset


# listens for signal from RIO that says to calibrate (change CALIBRATE = True)
def listen():
    print("this does nothing rn")

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
            print(fsnew)

            # adds data to a single list to make it less of a pain
            calibdatastruct.append(fsnew)

            # Yo dawg I heard you like pickles... (saving my variables in datafile)
            pickle.dump(calibdatastruct, datafile)
            datafile.close()

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
try:
    with open("calibrationdata.pkl", "rb") as datafile:
        saveddata = pickle.load(datafile)
        savedfs = saveddata[0]
except EOFError:
    savedfs = 2.5
    print("using default fs value (2.5)")

#datafile = open("calibrationdata.pkl", "rb")
#calibdata = pickle.load(datafile)
#datafile.close()

print("last focal value was " + str(savedfs))

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

    print(orangex1)
    ############################## UDP Stuff ###########################################

    cv2.imshow('orange1', output1orange)
    cv2.imshow('orange2', output2orange)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows
