#uses the same logic from Week 2 to identify the images 

import cv2
import numpy as np
from picamera2 import Picamera2

def setupCam():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

