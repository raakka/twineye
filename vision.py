import cv2
import numpy as np
import socket
import time
import math
import array
import pickle

############# Defining Variables and Tuning #############

<<<<<<< HEAD
b5142
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VidepCapture(1)
=======
ip = 10.13.11.22 # Change this number
port = 68756 # Change this number too

try
>>>>>>> c31f35f80d049bbed53e7376d39db308acd2d67b

# # # # # # YEET DEEP SPACE # # # # # #
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
#'--..__..--'               '-.____.-'                '--..__..-'

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
    cap1.set(cv2.CAP_PROP_EXPOSURE, exposure) # Disable auto exposure & white balance
    cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)# Disable autofocus
    print("    done.")
    return cap1

<<<<<<< HEAD
green_upper_colors = np.array([h, s, v])
green_lower_colors = np.array([h, s, v])

=======
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
    cap2.set(cv2.CAP_PROP_EXPOSURE, exposure) # Disable auto exposure & white balance
    cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)# Disable autofocus
    print("    done.")
    return cap2
>>>>>>> c31f35f80d049bbed53e7376d39db308acd2d67b

def greenprocess(image):
    hsvconv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask_green = cv2.inRange(image, green_lower, green_upper)
    mask_green = cv2.erode(mask_green, None, iterations = 2)
    mask_green = cv2.dilate(mask_green, None, iterations = 2)
    
    outputgreen = cv2.bitwise_and(image, image, mask = mask_green)
    ret, threshgreen = cv2.thresold(mask_green, 40, 255, 0)
    _, contgreen, _ = cv2.findContours(threshgreen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contgreen, outputorange
    
def orangeprocess(image):
    hsvconv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask_orange = cv2.inRange(image, orange_lower, orange_upper)
    mask_orange= cv2.erode(mask_orange, None, iterations = 2)
    mask_orange = cv2.dilate(mask_orange, None, iterations = 2)
    
    outputgreen = cv2.bitwise_and(image, image, mask = mask_green)
    ret, threshgreen = cv2.thresold(mask_green, 40, 255, 0)
    _, contorange, _ = cv2.findContours(threshorange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contorange, outputorange

def divbyzero(coord1, coord2):
    if (coord1 - coord2) == 0 or (coord2 - coord1) == 0:
        return True
    else:
        return False
    
def greencenter(contour):
    areaArray = []
    count = 1

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    largestcontour = sorteddata[0][1]
    secondlargestcontour = sorteddata[1][1]
    
    
    y1 = y1 + (h1/2)
    y2 = y2 + (h2/2)
    
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
    diff = None
    if xuno - xdos > 0:
        diff = (xuno - xdos)
        
    if xdos - xuno > 0:
        diff = (xdos - xuno)
        
def zcalc(diff, fs):
    if diff == 0 or diff == None:
        z = none
    else:
        z = (22 * fs) / diff
    return z
        
def distancenangle(z, offset):
    if z == none:
        dist = None
    else:
        dist == math.sqrt(math.pow(z, 2) + math.pow(offset, 2))
    return dist
    
# This function sends Data out via UDP
# At the moment it isn't functional so the ports and stuff still need some screwing around with
def send(data, port=50000, addr='239.192.1.100'):
    """send(data[, port[, addr]]) - multicasts a UDP datagram."""
    # Create the socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Make the socket multicast-aware, and set TTL.
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 20) # Change TTL (=20) to suit
    # Send the data
    s.sendto(data, (addr, port))
            
# Not functions...
green_upper = np.array([85, 200, 200] np.uint8)
green_lower = np.array([75, 120, 120], np.uint8)
orange_upper = np.array([5, 38, 255], np.uint8)
orange_lower = np.array([18, 20, 180], np.uint8)

global cap1
global cap2

cap1 = initCapture1()
cap2 = initCapture2()

#this is a really shitty way to do this, someone help fix pls

# This section calibrates if the user specifies / presses c on the keyboard
runcalibrate = input("press C to calibrate")

# Checks for C pressed
if (runcalibrate == 'c' or runcalibrate == 'C'):
    stopcalib = true
else:
    stopcalib = false
############################## Calibration Loop ####################################   
while (stopcalib == false):
    ret, img1 = cap1.read()
    ret, img2 = cap2.read()
    
    green1conts, output1green = greenprocess(img1)
    green2conts, output2green = greenprocess(img2)
    
    # Finding Contours
    if len(green1conts > 0 and green2conts > 0):
        cv2.drawContours(output1green, green1conts, -1, 255, 3)
        cont1max = max(green1conts, key = cv2.contourArea)
        xc1, yc1, wc1, hc1 = cv2.boundingRect(contmax1)
        camcoord1 = yc1
        
        cv2.drawContours(output2green, green2conts, -1, 255, 3)
        cont2max = max(green2conts, key = cv2.contourArea)
        xc2, yc2, wc2, hc2 = cv2.boundingRect(contmax2)
        camcoord2 = yc2
        
        # Stereo Calibration
        if(divbyzero(camcoord2, camcoord1) == False):
            fsnew = ((24 * (camcoord1 - camcoord2))/(22))
            testz = (22 * fsnew) / (camcoord1 - camcoord2)
            offsetnew = ((camcoord1 * testz) / fsnew) - 11
            print("This should equal 24 " + str(testz))
            
            with open('offsetdata.txt') as a:
                newTextoffset=a.read().replace(lastoffset, str(offsetnew)) # Replaces the previous calibration data
            with open('offsetdata.txt', "w") as a:
                    a.write(newTextoffset)
                    
            with open('fsdata.txt') as b:
                    newTextfs=b.read().replace(lastfs, str(fsnew))
            with open('fsdata.txt', "w") as b:
                    b.write(newTextfs)
                    
            print("Calibration Complete")
            stopcalib = true
        else:
            stopcalib = false 

############################## Detection Loop ####################################

while True:
    ret, img1 = cap1
    ret, img2 = cap2
   
    green1conts, output1green = greenprocess(img1)
    green2conts, output2green = greenprocess(img2)
    orange1conts, output1orange = orangeprocess(img1)
    orange2conts, output2orange = orangeprocess(img2)
    
    if len(green1conts >= 2 and green2conts >= 2):
        greenx1, greeny1 = greencenter(green1conts)
        greenx2, greeny2 = greencenter(green2conts)
        
    if len(orange1conts != 0):
        o1 = max(orange1conts, key = cv2.contourArea)
        orangex1, orangey1, orangew1, orangeh1 = cv2.boundingRectangle(o1)
    else:
        orangex1, orangey1, orangew1, orangeh1 = None
    
    if len(orange2conts != 0):
        o2 = max(orange2conts, key = cv2.contourArea)
        orangex2, orangey2, orangew2, orangeh2 = cv2.boundingRectangle(o2)
    else:
        orangex2, orangey2, orangew2, orangeh2 = None
        
############################## Stereo Calculations #################################

    orangediff = xdiff(orangex1, orangex2) 
    greendiff = xdiff(greenx1, greenx2)
    
    orangez = zcalc(orangediff, savedfs)
    greenz = zcalc(greendiff, savedfs)
    
    orangedist, orangeangle = distancenangle(orangez, savedoffset)
    greendist, greenangle = distancenangle(greenz, savedoffset)
    
    send(str(greendist).encode, port, ip) #the ip should be fixed but the port changes, any ideas how to stop it (keep fixed port number???)
    
    k = cv2.waitkey(30) & 0xff
    if k = 27:
        break
        
cap1.release()
cap2.release()
cv2.destroyAllWindows
