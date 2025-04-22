#uses face_recognition module, should be used if face_detection can be installed

#1. use pip to install
#2. use git to clone the directory then install 
# git clone https://github.com/ageitgey/face_recognition.git
# cd face_recognition
# pip install .

import cv2
import face_recognition
import numpy as np
import os

known_face_dir = '/home/pi/Downloads/faces/'  # Update with your known face directory path
known_face_encoding = []
known_face_names = []

for filename in os.listdir(known_face_dir):
    if filename.endswith(('.jpg', '.png')):
        image = face_recognition.load_image_file(os.path.join(known_face_dir, filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encoding.append(encoding)
        known_face_names.append(filename.split('.')[0])


def setupCam():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    return picam2

cam = setupCam()

try:
    while True:
        frame = cam.capture_array()
        
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgbFrame)
        face_encodings = face_recognition.face_encodings(rgbFrame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding, tolerance=0.6)

            name = "Unknown"

            faceDistances = face_recognition.face_distance(known_face_encoding, face_encoding)

            if len(faceDistances) > 0:
                bestMatchIndex = np.argmin(faceDistances)

                if matches[bestMatchIndex]:
                    name = known_face_names[bestMatchIndex]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top, -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        cv2.imshow('Face Recognition', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    cam.stop()
    print("Program stopped by user")