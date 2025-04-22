#this uses basic template matching for image detection (not working)

import cv2
import numpy as np
import os
from picamera2 import Picamera2

def setupCam():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

templateDirectory = "/home/pi/Downloads/data/"  # Update with your template directory path

templateFiles = [f for f in os.listdir(templateDirectory) if f.endswith('.jpg') or f.endswith('.png')]
templates = []

for templateFile in templateFiles:
    templateImg = cv2.imread(os.path.join(templateDirectory, templateFile))
    templateGray = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
    templates.append({
        'name': templateFile,
        'gray': templateGray,
        'w': templateGray.shape[1],
        'h': templateGray.shape[0]
    })

try:
    cam = setupCam()

    while True:
        frame = cam.capture_array()

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frameHSV, (0, 0, 0), (180, 255, 255))

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameGray = cv2.bitwise_and(frameGray, frameGray, mask=mask)

        for template in templates:
            result = cv2.matchTemplate(frameGray, template['gray'], cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

            if maxVal > 0.8:  # Adjust threshold as needed
                topLeft = maxLoc
                bottomRight = (topLeft[0] + template['w'], topLeft[1] + template['h'])
                cv2.rectangle(frame, topLeft, bottomRight, (0, 255, 0), 2)
                cv2.putText(frame, template['name'], (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('cam feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    cam.stop()
    print("Program stopped by user")