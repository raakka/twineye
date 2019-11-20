#!/usr/bin/env python
"""This script is for a GUI that allows you to tune the HSV thresholding values for use in a vision pipeline.
The minimum and maximum HSV values are saved in a JSON file called config.json.
"""

import os
import json
from tkinter import *
import numpy as np
import cv2
from PIL import Image
from PIL import ImageTk

__author__ = "SumiGovindaraju"
__copyright__ = "Copyright 2018, SumiGovindaraju"
__credits__ = ["SumiGovindaraju"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "SumiGovindaraju"
__status__ = "Development"

def pipeline(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10)))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    return opening

def saveHSVToFileAndClose():
    file = open("config.json", "w")

    jsonObj = {
        'min': {
            'hue': str(hsv_min[0]),
            'saturation': str(hsv_min[1]),
            'value': str(hsv_min[2])
        },
        'max': {
            'hue': str(hsv_max[0]),
            'saturation': str(hsv_max[1]),
            'value': str(hsv_max[2])
        }
    }

    file.write(u"" + json.dumps(jsonObj))

    root.quit()

def loadHSVFromFile(key):
    try:
        with open("config.json") as json_data:
            data = json.load(json_data)
            return np.array([data[key]["hue"], data[key]["saturation"], data[key]["value"]], dtype=np.uint8)
    except IOError as e:
        return np.array([90, 128, 128], dtype=np.uint8)

def addPixelHSV():
    if selectedPixel[0] < 0 or selectedPixel[1] < 0:
        return

    hsv = cv2.cvtColor(app.frame, cv2.COLOR_BGR2HSV)
    selectedHSV = hsv[selectedPixel[1], selectedPixel[0]]

    if int(selectedHSV[0]) < hsv_min[0]:
        app.huemin.set(int(selectedHSV[0]) - 10)
    if int(selectedHSV[1]) < hsv_min[1]:
        app.satmin.set(int(selectedHSV[1]) - 10)
    if int(selectedHSV[2]) < hsv_min[2]:
        app.valmin.set(int(selectedHSV[2]) - 10)
    if int(selectedHSV[0]) > hsv_max[0]:
        app.huemax.set(int(selectedHSV[0]) + 10)
    if int(selectedHSV[1]) > hsv_max[1]:
        app.satmax.set(int(selectedHSV[1]) + 10)
    if int(selectedHSV[2]) > hsv_max[2]:
        app.valmax.set(int(selectedHSV[2]) + 10)

def subtractPixelHSV():
    if selectedPixel[0] < 0 or selectedPixel[1] < 0:
        return

    hsv = cv2.cvtColor(app.frame, cv2.COLOR_BGR2HSV)
    selectedHSV = hsv[selectedPixel[1], selectedPixel[0]]

    if abs(selectedHSV[0] - hsv_min[0]) < abs(selectedHSV[0] - hsv_max[0]):
        app.huemin.set(selectedHSV[0] + 10)
    else:
        app.huemax.set(selectedHSV[0] - 10)

    if abs(selectedHSV[1] - hsv_min[1]) < abs(selectedHSV[1] - hsv_max[1]):
        app.satmin.set(selectedHSV[1] + 10)
    else:
        app.satmax.set(selectedHSV[1] - 10)

    if abs(selectedHSV[2] - hsv_min[2]) < abs(selectedHSV[2] - hsv_max[2]):
        app.valmin.set(selectedHSV[2] + 10)
    else:
        app.valmax.set(selectedHSV[2] - 10)

def selectPixel(event):
    selectedPixel[0] = event.x
    selectedPixel[1] = event.y

    cv2.namedWindow("Pixel Selected")
    b,g,r = app.frame[selectedPixel[1], selectedPixel[0]]
    pixel = np.zeros((50, 50, 3), np.uint8)
    pixel[:] = [b, g, r]
    cv2.imshow("Pixel Selected", pixel)

hsv_min = loadHSVFromFile("min")
hsv_max = loadHSVFromFile("max")
selectedPixel = [-1, -1]

class TunerWindow(Frame):
    def video_loop(self):
        if self.valmax.winfo_exists() == 1:
            hsv_min[0] = self.huemin.get()
            hsv_min[1] = self.satmin.get()
            hsv_min[2] = self.valmin.get()
            hsv_max[0] = self.huemax.get()
            hsv_max[1] = self.satmax.get()
            hsv_max[2] = self.valmax.get()

        retval, self.frame = self.videocapture.read()
        self.frame = cv2.resize(self.frame, (0,0), fx=0.4, fy=0.4)
        mask = pipeline(self.frame)
        if retval:
            cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA) if self.showRawPhoto.get() else mask
            currentImg = Image.fromarray(cv2image)
            currentImgTK = ImageTk.PhotoImage(image=currentImg)
            self.imgpanel.currentImgTK = currentImgTK
            self.imgpanel.config(image=currentImgTK)

    def __init__(self, master):
        self.tk = master

        # Camera settings and such
        # Play around with these to get faster camera feed
        self.videocapture = cv2.VideoCapture(0)
        self.videocapture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        default_exposure = self.videocapture.get(cv2.CAP_PROP_EXPOSURE)
        custom_exposure = 1 #can be changed
        self.videocapture.set(cv2.CAP_PROP_EXPOSURE, default_exposure)

        self.tk.title("Vision Pipeline Tuning")
        self.tk.protocol("WM_DELETE_WINDOW", saveHSVToFileAndClose)

        self.imgpanel = Label(self.tk)
        self.imgpanel.bind("<Button-1>", selectPixel)
        self.imgpanel.pack()

        self.buttonpanel = Label(self.tk)
        self.buttonpanel.pack()

        self.showRawPhoto = IntVar()
        toggleMask = Checkbutton(self.tk, text="Toggle Mask", onvalue=False, offvalue=True, variable=self.showRawPhoto)
        toggleMask.pack(in_=self.buttonpanel, side=LEFT)

        addPixel = Button(self.tk, text="Add Pixel", command=addPixelHSV)
        addPixel.pack(in_=self.buttonpanel, side=LEFT)

        subtractPixel = Button(self.tk, text="Subtract Pixel", command=subtractPixelHSV)
        subtractPixel.pack(in_=self.buttonpanel, side=RIGHT)

        self.sliderpanel = Label(self.tk)
        self.sliderpanel.pack()

        hueminLabel = Label(self.tk, text="Hue Min:")
        hueminLabel.pack()
        self.huemin = Scale(master, from_=0, to=180, orient=HORIZONTAL)
        self.huemin.set(hsv_min[0])
        self.huemin.pack()

        huemaxLabel = Label(self.tk, text="Hue Max:")
        huemaxLabel.pack()
        self.huemax = Scale(master, from_=0, to=180, orient=HORIZONTAL)
        self.huemax.set(hsv_max[0])
        self.huemax.pack()

        satminLabel = Label(self.tk, text="Sat Min:")
        satminLabel.pack()
        self.satmin = Scale(master, from_=0, to=255, orient=HORIZONTAL)
        self.satmin.set(hsv_min[1])
        self.satmin.pack()

        satmaxLabel = Label(self.tk, text="Sat Max:")
        satmaxLabel.pack()
        self.satmax = Scale(master, from_=0, to=255, orient=HORIZONTAL)
        self.satmax.set(hsv_max[1])
        self.satmax.pack()

        valminLabel = Label(self.tk, text="Val Min:")
        valminLabel.pack()
        self.valmin = Scale(master, from_=0, to=255, orient=HORIZONTAL)
        self.valmin.set(hsv_min[2])
        self.valmin.pack()

        valmaxLabel = Label(self.tk, text="Val Max:")
        valmaxLabel.pack()
        self.valmax = Scale(master, from_=0, to=255, orient=HORIZONTAL)
        self.valmax.set(hsv_max[2])
        self.valmax.pack()

root = Tk()
app = TunerWindow(root)

if __name__ == "__main__":
    while True:
        app.video_loop()

        app.update_idletasks()
        app.update()
