from doinit import HandDetector
import cv2
import math
import numpy as np
from ctypes import cast , POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices, volume, interface = None, None ,None
def volume_init():
    global devices
    global volume
    global interface

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # print(volume.GetVolumeRange())



hand_detector = HandDetector(min_dtct_cnfdnc=0.7)
camera = cv2.VideoCapture(0)

volume_init()

while True:
    status, img = camera.read()
    hand_landmarks = hand_detector.findHandLMs(img=img, draw=True)

    if len(hand_landmarks) != 0 :
        x1, y1 = hand_landmarks[4][1], hand_landmarks[4][2]
        x2, y2 = hand_landmarks[8][1], hand_landmarks[8][2]

        length = math.hypot(x2-x1, y2-y1)
        print(length)

        volume_value = np.interp(length, [50, 250], [-65.25, 0.0])
        volume.SetMasterVolumeLevel(volume_value, None)


        cv2.circle(img, (x1,y1), 16, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 16, (255,0,255), cv2.FILLED)

    cv2.imshow('volume', img)
    cv2.waitKey(1)
