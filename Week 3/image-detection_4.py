#Detects symbols and images using tensorflow

#install tflite-runtime, then unzip the savedmodel files

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

IN1, IN2 = 23, 24
IN3, IN4 = 27, 22
ENA, ENB = 13, 12

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

interpreter = tflite.Interpreter(model_path='pi/Downloads/Week_3/model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("/path/to/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def setupCam():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

cam = setupCam()

try:
    while True:
        frame = cam.capture_array()

        inputShape = input_details[0]["shape"]
        image = cv2.resize(frame, (inputShape[1], inputShape[2]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]["index"], image)

        interpreter.invoke()

        outputData = interpreter.get_tensor(output_details[0]["index"])
        probabilities = np.squeeze(outputData)

        predictedClassIdx = np.argmax(probabilities)
        predictedLabel = labels[predictedClassIdx]
        confidence = probabilities[predictedClassIdx]

        text = f"{predictedLabel}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Teachable Machine", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nProgram stopped by user")